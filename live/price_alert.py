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


def build_message(stats: dict, candle: dict, signal: dict) -> str:
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
        f"As of {now}"
    )


def send_one():
    stats = fetch_price_stats(SYMBOL)
    closed = fetch_15m_klines(SYMBOL)
    candle = candle_detail(closed)
    signal = ema_signal(closed)
    msg = build_message(stats, candle, signal)
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
