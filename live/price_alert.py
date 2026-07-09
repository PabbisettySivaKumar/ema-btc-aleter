"""
Price + EMA9/21 crossover alerter for BTC/USDT — the strategy foundation.

Every 15 minutes it sends to Telegram:
  - current price, 15m & 24h change, 24h range
  - the last closed 15m candle's OHLC + move vs the previous candle
  - EMA9/21 signal state, flagging loudly when a crossover just fired

Strategy: bare EMA9/21 crossover on the 15m timeframe (no other filters).
You review each alert and decide — nothing is auto-executed.

Data source: data-api.binance.vision — Binance's public market-data mirror.
It serves the GLOBAL binance.com order book, so the price matches the Binance
app's BTCUSDT to the cent, and it is not geo-blocked on GitHub Actions runners.

Run once (cron / GitHub Actions):
    python live/price_alert.py --once
Run continuously (VPS / your machine), pinging aligned to :00/:15/:30/:45:
    python live/price_alert.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alerts.notifier import send_telegram

IST = timezone(timedelta(hours=5, minutes=30))   # India Standard Time (no DST)

SYMBOL = "BTCUSDT"
BINANCE_24HR_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

# --- Strategy: EMA9/21 crossover on the 15m timeframe ---
EMA_FAST = 9
EMA_SLOW = 21

POLL_INTERVAL_SECONDS = 15 * 60   # 15 minutes
POLL_ALIGN_OFFSET_SECONDS = 5     # a few seconds after the :00/:15/:30/:45 mark

# --- Signal portfolio tracker (informational, compounding) ---
# A hypothetical $1,000 that BUYS on each bullish EMA9/21 crossover and SELLS on
# the next bearish one, compounding the proceeds into the next trade. Marked to
# market on every alert. NOTE: backtests show this round-trip loses on 15m — it's
# here to visualize what the signal does, not a strategy to fund.
START_CAPITAL_USD = 1000.0
BINANCE_FEE_PCT = 0.1     # Binance India spot fee — charged on BOTH the buy and the sell
TRADE_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_state.json")


def fetch_price_stats(symbol: str) -> dict:
    """Current price + 24h change for `symbol` from the global Binance mirror."""
    resp = requests.get(BINANCE_24HR_URL, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    d = resp.json()
    return {
        "price": float(d["lastPrice"]),
        "change_pct": float(d["priceChangePercent"]),
        "high": float(d["highPrice"]),
        "low": float(d["lowPrice"]),
    }


def fetch_15m_klines(symbol: str, limit: int = 120) -> list:
    """Recent 15m candles with the still-forming one dropped, so the LAST
    element is the most recent CLOSED candle. `limit` >> EMA_SLOW for warmup."""
    resp = requests.get(
        BINANCE_KLINES_URL,
        params={"symbol": symbol, "interval": "15m", "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()[:-1]   # drop the in-progress candle


def candle_detail(closed: list) -> dict:
    """OHLC of the most recent CLOSED 15m candle, its intra-candle move, and
    the close-to-close difference vs the previous candle."""
    last, prev = closed[-1], closed[-2]
    o, h, l, c = float(last[1]), float(last[2]), float(last[3]), float(last[4])
    prev_close = float(prev[4])
    open_ist = datetime.fromtimestamp(last[0] / 1000, tz=IST)
    close_ist = open_ist + timedelta(minutes=15)   # interval end (open + 15m), clean :15/:30/:45/:00
    return {
        "open": o, "high": h, "low": l, "close": c,
        "change_pct": (c / o - 1) * 100,          # intra-candle move (open -> close)
        "diff_abs": c - prev_close,               # vs previous candle's close
        "diff_pct": (c / prev_close - 1) * 100,
        "open_ist": open_ist, "close_ist": close_ist,
    }


def ema_signal(closed: list) -> dict:
    """EMA9/21 state on the closed 15m candles. Reports the current regime
    (fast vs slow) and whether a crossover just fired on the last closed bar.
    EMA uses ewm(span, adjust=False) to match the rest of the project."""
    closes = pd.Series([float(r[4]) for r in closed])
    ema_fast = closes.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = closes.ewm(span=EMA_SLOW, adjust=False).mean()
    f_now, s_now = ema_fast.iloc[-1], ema_slow.iloc[-1]
    f_prev, s_prev = ema_fast.iloc[-2], ema_slow.iloc[-2]

    crossover = None
    if f_prev <= s_prev and f_now > s_now:
        crossover = "bullish"
    elif f_prev >= s_prev and f_now < s_now:
        crossover = "bearish"

    return {
        "ema_fast": float(f_now),
        "ema_slow": float(s_now),
        "state": "bullish" if f_now > s_now else "bearish",
        "crossover": crossover,
    }


def build_signal_section(signal: dict) -> str:
    trend = ("🟢 fast > slow (bullish)" if signal["state"] == "bullish"
             else "🔴 fast < slow (bearish)")
    if signal["crossover"] == "bullish":
        headline = "🟢🟢 BULLISH CROSSOVER — EMA9 crossed ABOVE EMA21! Possible LONG."
    elif signal["crossover"] == "bearish":
        headline = "🔴🔴 BEARISH CROSSOVER — EMA9 crossed BELOW EMA21! Exit / caution."
    else:
        headline = "• No new crossover this candle."
    return (
        f"📊 Signal (EMA{EMA_FAST}/{EMA_SLOW}, 15m):\n"
        f"  EMA{EMA_FAST}:  ${signal['ema_fast']:,.2f}\n"
        f"  EMA{EMA_SLOW}: ${signal['ema_slow']:,.2f}\n"
        f"  Trend: {trend}\n"
        f"  {headline}"
    )


def find_crossovers(closed: list) -> list:
    """Every EMA9/21 crossover in the closed candles, chronological. Each item:
    {ts, price, direction, time_ist}. ts = candle open_time (ms); the signal
    fires on the candle CLOSE, so `price` is that candle's close."""
    closes = pd.Series([float(r[4]) for r in closed])
    ef = closes.ewm(span=EMA_FAST, adjust=False).mean()
    es = closes.ewm(span=EMA_SLOW, adjust=False).mean()
    fast_above = ef > es
    out = []
    for i in range(1, len(closed)):
        if fast_above.iloc[i] and not fast_above.iloc[i - 1]:
            direction = "bullish"
        elif not fast_above.iloc[i] and fast_above.iloc[i - 1]:
            direction = "bearish"
        else:
            continue
        ts = int(closed[i][0])
        close_ist = datetime.fromtimestamp(ts / 1000, tz=IST) + timedelta(minutes=15)
        out.append({"ts": ts, "price": float(closed[i][4]), "direction": direction,
                    "time_ist": close_ist.strftime("%d %b %H:%M IST")})
    return out


def load_or_init_trade_state(crossovers: list):
    """Read the compounding signal-portfolio state, or start it fresh with
    $1,000 in cash. On init, all currently-visible crossovers are marked as
    already-seen so trading only begins on the NEXT crossover.
    Returns (state, is_new)."""
    if os.path.exists(TRADE_STATE_PATH):
        try:
            with open(TRADE_STATE_PATH) as f:
                st = json.load(f)
            if "balance" in st:
                return st, False
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # missing/corrupt -> re-initialize below
    now = datetime.now(tz=IST)
    st = {
        "started_ist": now.strftime("%Y-%m-%d %H:%M IST"),
        "balance": START_CAPITAL_USD,      # cash on hand when flat
        "in_position": False,
        "entry_price": None,
        "entry_time_ist": None,
        "btc": 0.0,
        "cost_basis": 0.0,                 # $ put into the current open trade
        "last_crossover_ts": crossovers[-1]["ts"] if crossovers else 0,
        "num_trades": 0,
        "wins": 0,
        "last_trade_pct": None,
        "last_exit_ist": None,
    }
    return st, True


def process_crossovers(st: dict, crossovers: list) -> bool:
    """Apply any crossovers newer than last_crossover_ts: buy on bullish (when
    flat), sell on bearish (when in a position), compounding proceeds. Binance
    fees taken on both sides. Returns True if the state changed."""
    fee = BINANCE_FEE_PCT / 100
    changed = False
    for x in crossovers:
        if x["ts"] <= st["last_crossover_ts"]:
            continue
        if x["direction"] == "bullish" and not st["in_position"]:
            capital = st["balance"]
            st.update(in_position=True, entry_price=x["price"],
                      btc=capital * (1 - fee) / x["price"],   # buy fee reduces BTC received
                      cost_basis=capital, entry_time_ist=x["time_ist"])
        elif x["direction"] == "bearish" and st["in_position"]:
            proceeds = st["btc"] * x["price"] * (1 - fee)     # sell fee
            st.update(in_position=False, balance=proceeds,
                      num_trades=st["num_trades"] + 1,
                      wins=st["wins"] + (1 if proceeds > st["cost_basis"] else 0),
                      last_trade_pct=(proceeds / st["cost_basis"] - 1) * 100,
                      last_exit_ist=x["time_ist"],
                      entry_price=None, btc=0.0, cost_basis=0.0, entry_time_ist=None)
        st["last_crossover_ts"] = x["ts"]
        changed = True
    return changed


def save_trade_state(st: dict):
    try:
        with open(TRADE_STATE_PATH, "w") as f:
            json.dump(st, f, indent=2)
    except OSError as e:
        print(f"warning: could not write trade state: {e}")


def build_trade_section(st: dict, current_price: float) -> str:
    start = START_CAPITAL_USD
    if st["in_position"]:
        net_now = st["btc"] * current_price * (1 - BINANCE_FEE_PCT / 100)  # value if sold now
        unreal_abs = net_now - st["cost_basis"]
        unreal_pct = (net_now / st["cost_basis"] - 1) * 100
        arrow = "🟢" if unreal_abs >= 0 else "🔴"
        head = f"💼 Signal portfolio: ${net_now:,.2f} ({(net_now/start-1)*100:+.2f}% since {st['started_ist']})"
        body = (
            f"  📈 In a trade — bought on bull cross {st['entry_time_ist']}\n"
            f"  {st['btc']:.6f} BTC @ ${st['entry_price']:,.2f} (cost ${st['cost_basis']:,.2f})\n"
            f"  Now ${current_price:,.2f} → worth ${net_now:,.2f} net of fees\n"
            f"  Unrealized: {arrow} {unreal_pct:+.2f}% (${unreal_abs:+,.2f})"
        )
    else:
        bal = st["balance"]
        head = f"💼 Signal portfolio: ${bal:,.2f} ({(bal/start-1)*100:+.2f}% since {st['started_ist']})"
        if st["num_trades"] == 0:
            body = "  💵 In cash — waiting for the first bullish crossover to buy."
        else:
            la = "🟢" if (st["last_trade_pct"] or 0) >= 0 else "🔴"
            body = (
                f"  💵 In cash since {st['last_exit_ist']} (sold on bear cross)\n"
                f"  Last trade: {la} {st['last_trade_pct']:+.2f}% net\n"
                f"  Waiting for next bullish crossover."
            )
    tail = f"  Trades closed: {st['num_trades']}"
    if st["num_trades"]:
        tail += f" (wins {st['wins']}/{st['num_trades']})"
    return f"{head}\n{body}\n{tail}"


def build_message(stats: dict, candle: dict, signal: dict, trade_state: dict) -> str:
    now = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M IST")
    arrow_24h = "🟢" if stats["change_pct"] >= 0 else "🔴"
    arrow_15m = "🟢" if candle["change_pct"] >= 0 else "🔴"
    arrow_diff = "🟢" if candle["diff_abs"] >= 0 else "🔴"
    candle_range = (
        f"{candle['open_ist'].strftime('%d %b %H:%M')}"
        f"–{candle['close_ist'].strftime('%H:%M')} IST"
    )
    return (
        f"💰 {SYMBOL} price update\n"
        f"Price: ${stats['price']:,.2f}\n"
        f"15m:   {arrow_15m} {candle['change_pct']:+.2f}%\n"
        f"24h:   {arrow_24h} {stats['change_pct']:+.2f}%\n"
        f"24h high/low: ${stats['high']:,.2f} / ${stats['low']:,.2f}\n"
        f"\n"
        f"Last 15m candle ({candle_range}):\n"
        f"  O: ${candle['open']:,.2f}   H: ${candle['high']:,.2f}\n"
        f"  L: ${candle['low']:,.2f}   C: ${candle['close']:,.2f}\n"
        f"  vs prev candle: {arrow_diff} {candle['diff_abs']:+,.2f} ({candle['diff_pct']:+.2f}%)\n"
        f"\n"
        f"{build_signal_section(signal)}\n"
        f"\n"
        f"{build_trade_section(trade_state, stats['price'])}\n"
        f"\n"
        f"As of {now}"
    )


def send_one():
    stats = fetch_price_stats(SYMBOL)
    closed = fetch_15m_klines(SYMBOL, limit=300)
    candle = candle_detail(closed)
    signal = ema_signal(closed)
    crossovers = find_crossovers(closed)
    trade_state, is_new = load_or_init_trade_state(crossovers)
    if process_crossovers(trade_state, crossovers) or is_new:
        save_trade_state(trade_state)
    msg = build_message(stats, candle, signal, trade_state)
    print(msg)
    send_telegram(msg)


def sleep_until_next_slot():
    now = time.time()
    slot = POLL_INTERVAL_SECONDS
    next_slot = (int(now // slot) + 1) * slot + POLL_ALIGN_OFFSET_SECONDS
    wait = next_slot - now
    if wait > 0:
        print(f"Sleeping {int(wait)}s until next 15-min slot ...")
        time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="Send a single price alert then exit (for cron / GitHub Actions).")
    args = parser.parse_args()

    if args.once:
        send_one()
        return

    print(f"Price alerter started for {SYMBOL} — pinging every 15 min.")
    while True:
        try:
            send_one()
        except Exception as e:
            print(f"error: {e}")
        sleep_until_next_slot()


if __name__ == "__main__":
    main()
