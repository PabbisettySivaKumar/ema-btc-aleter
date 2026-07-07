"""
Dual-track live Telegram alerter for BTC/USDT.

Runs two independent strategy tracks in parallel and pings Telegram:
  - On entry crossover
  - Price update every 15 min while a trade is open
  - +2% target reached (consider taking profit — you decide manually)
  - Track A only: -5% stop hit
  - Opposite crossover (position considered "ended")

TRACK A (data winner):
  1h timeframe, 20/50 EMA, 4h 200 EMA macro filter, volume filter,
  -5% stop, +2% target notification, exit on opposite 1h crossover.

TRACK B (user preference):
  15m timeframe, 9/21 EMA, 4h 200 EMA macro filter, volume filter,
  NO stop loss, +2% target notification, exit on opposite 15m crossover.

Never places orders. Alerts only. You review each alert and decide.

Setup:
  1. pip install requests pandas python-dotenv numpy
  2. Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
  3. python live/dual_track_alerter.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alerts.notifier import send_telegram

SYMBOL = "BTCUSDT"
# Use binance.us because GitHub Actions runners are on US IPs and
# binance.com returns HTTP 451 (geo-restricted) from those. binance.us
# serves the same symbols with prices that track binance.com within ~0.05%.
BINANCE_KLINES_URL = "https://api.binance.us/api/v3/klines"
BINANCE_TICKER_URL = "https://api.binance.us/api/v3/ticker/price"


def fetch_ticker_price(symbol: str) -> float:
    resp = requests.get(BINANCE_TICKER_URL, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    return float(resp.json()["price"])

# --- Shared filters ---
MACRO_TF = "4h"
MACRO_EMA = 200
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.2
TARGET_PCT = 2.0
POLL_INTERVAL_SECONDS = 15 * 60          # 15 minutes
POLL_ALIGN_OFFSET_SECONDS = 20           # start 20s after :00/:15/:30/:45 to let candle close

# --- Track configs ---
TRACK_A = {
    "name": "TRACK-A (1h 20/50, -5% stop)",
    "short": "A",
    "tf": "1h",
    "tf_minutes": 60,
    "ema_fast": 20,
    "ema_slow": 50,
    "stop_pct": 5.0,
    "state_file": "live/state_track_a.json",
    "always_pulse": False,   # only alert on real events
}
TRACK_B = {
    "name": "TRACK-B (15m 9/21, no stop)",
    "short": "B",
    "tf": "15m",
    "tf_minutes": 15,
    "ema_fast": 9,
    "ema_slow": 21,
    "stop_pct": None,      # no stop
    "state_file": "live/state_track_b.json",
    "always_pulse": True,    # 15-min market-pulse ping even when flat
}


# =====================================================================
# Data fetch
# =====================================================================

def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch the most recent `limit` candles ending at the last CLOSED candle."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time")
    # Drop the in-progress candle (the last row's close_time is in the future)
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] < now]
    return df


# =====================================================================
# Indicators
# =====================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_indicators_for_track(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg["ema_fast"])
    out["ema_slow"] = ema(out["close"], cfg["ema_slow"])
    out["volume_sma"] = out["volume"].rolling(VOLUME_SMA_PERIOD).mean()
    return out


def get_macro_state(symbol: str) -> tuple[float, float]:
    """Return (last_4h_close, 4h_200_ema) from last CLOSED 4h bar."""
    df = fetch_klines(symbol, MACRO_TF, limit=MACRO_EMA + 50)
    df["ema_macro"] = ema(df["close"], MACRO_EMA)
    last = df.iloc[-1]
    return float(last["close"]), float(last["ema_macro"])


# =====================================================================
# State persistence
# =====================================================================

def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"open": False, "entry_price": None, "entry_time": None,
                "target_hit": False, "last_signal_time": None,
                "last_update_bar_time": None}
    with open(path) as f:
        return json.load(f)


def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


# =====================================================================
# Signal + exit checks
# =====================================================================

def check_entry_signal(df: pd.DataFrame, macro_close: float, macro_ema: float) -> bool:
    """True if the last CLOSED bar shows a valid crossover entry."""
    if len(df) < 3:
        return False
    row, prev = df.iloc[-1], df.iloc[-2]
    if np.isnan(row["ema_fast"]) or np.isnan(row["ema_slow"]) or np.isnan(row["volume_sma"]):
        return False
    crossed_up = (prev["ema_fast"] <= prev["ema_slow"]) and (row["ema_fast"] > row["ema_slow"])
    macro_ok = macro_close > macro_ema
    volume_ok = row["volume"] > VOLUME_MULTIPLIER * row["volume_sma"]
    return bool(crossed_up and macro_ok and volume_ok)


def check_exit_crossover(df: pd.DataFrame) -> bool:
    """True if the fast EMA has just crossed BELOW the slow EMA (trend ended)."""
    if len(df) < 3:
        return False
    row, prev = df.iloc[-1], df.iloc[-2]
    return bool((prev["ema_fast"] >= prev["ema_slow"]) and (row["ema_fast"] < row["ema_slow"]))


# =====================================================================
# Alerts
# =====================================================================

def fmt_price(p: float) -> str:
    return f"${p:,.2f}"


def alert_entry(cfg: dict, entry_price: float, ts: pd.Timestamp):
    stop_line = ""
    if cfg["stop_pct"] is not None:
        stop_price = entry_price * (1 - cfg["stop_pct"] / 100.0)
        stop_line = f"\nSafety stop:  {fmt_price(stop_price)}  (-{cfg['stop_pct']:.0f}%)"
    target_price = entry_price * (1 + TARGET_PCT / 100.0)
    msg = (
        f"🟢 ENTRY SIGNAL — {cfg['name']}\n"
        f"Symbol: {SYMBOL}\n"
        f"Bar close time: {ts}\n"
        f"Suggested entry: {fmt_price(entry_price)}\n"
        f"Target (+{TARGET_PCT:.0f}%): {fmt_price(target_price)}"
        f"{stop_line}\n"
        f"⚠️ Manual review before placing. Not auto-executed."
    )
    send_telegram(msg)
    print(f"[{cfg['short']}] ENTRY ALERT sent @ {entry_price}")


def alert_price_update(cfg: dict, entry_price: float, current_price: float,
                       target_hit: bool):
    pnl_pct = (current_price / entry_price - 1) * 100.0
    status = "🎯 TARGET REACHED — you can hold or sell." if target_hit else "…in trade, watching."
    msg = (
        f"📈 {cfg['short']} update — {SYMBOL}\n"
        f"Entry: {fmt_price(entry_price)}\n"
        f"Now:   {fmt_price(current_price)}\n"
        f"P&L:   {pnl_pct:+.2f}%\n"
        f"{status}"
    )
    send_telegram(msg)
    print(f"[{cfg['short']}] update sent: {pnl_pct:+.2f}%")


def alert_target_hit(cfg: dict, entry_price: float, current_price: float):
    pnl_pct = (current_price / entry_price - 1) * 100.0
    msg = (
        f"🎯 {cfg['short']} TARGET HIT (+{TARGET_PCT:.0f}%) — {SYMBOL}\n"
        f"Entry: {fmt_price(entry_price)}\n"
        f"Now:   {fmt_price(current_price)} ({pnl_pct:+.2f}%)\n"
        f"Consider taking profit. Position will keep being tracked until opposite crossover."
    )
    send_telegram(msg)
    print(f"[{cfg['short']}] TARGET HIT")


def alert_stop_hit(cfg: dict, entry_price: float, current_price: float):
    pnl_pct = (current_price / entry_price - 1) * 100.0
    msg = (
        f"🛑 {cfg['short']} STOP HIT ({-cfg['stop_pct']:.0f}%) — {SYMBOL}\n"
        f"Entry: {fmt_price(entry_price)}\n"
        f"Now:   {fmt_price(current_price)} ({pnl_pct:+.2f}%)\n"
        f"Cut the position now. Position tracker reset."
    )
    send_telegram(msg)
    print(f"[{cfg['short']}] STOP HIT")


def alert_market_pulse(cfg: dict, current_price: float, fast_ema: float,
                       slow_ema: float, macro_close: float, macro_ema: float):
    fast_vs_slow = "fast > slow ✅" if fast_ema > slow_ema else "fast < slow ❌"
    macro_state = "macro up ✅" if macro_close > macro_ema else "macro DOWN ❌"
    msg = (
        f"⏱ {cfg['short']} pulse — {SYMBOL}\n"
        f"Price: {fmt_price(current_price)}\n"
        f"{cfg['ema_fast']} EMA: {fmt_price(fast_ema)}\n"
        f"{cfg['ema_slow']} EMA: {fmt_price(slow_ema)}  ({fast_vs_slow})\n"
        f"4h 200 EMA: {fmt_price(macro_ema)}  ({macro_state})\n"
        f"Status: FLAT — no open position."
    )
    send_telegram(msg)
    print(f"[{cfg['short']}] pulse sent (flat)")


def alert_trend_end(cfg: dict, entry_price: float, current_price: float):
    pnl_pct = (current_price / entry_price - 1) * 100.0
    msg = (
        f"🔴 {cfg['short']} TREND ENDED — {SYMBOL}\n"
        f"(fast EMA crossed below slow EMA)\n"
        f"Entry: {fmt_price(entry_price)}\n"
        f"Now:   {fmt_price(current_price)} ({pnl_pct:+.2f}%)\n"
        f"Position tracker reset — waiting for next entry signal."
    )
    send_telegram(msg)
    print(f"[{cfg['short']}] TREND END sent")


# =====================================================================
# Per-track cycle
# =====================================================================

def process_track(cfg: dict, macro_close: float, macro_ema: float):
    state = load_state(cfg["state_file"])

    df = fetch_klines(SYMBOL, cfg["tf"], limit=max(cfg["ema_slow"] * 5, 200))
    df = compute_indicators_for_track(df, cfg)
    if df.empty:
        print(f"[{cfg['short']}] no data — skipping")
        return

    last = df.iloc[-1]
    last_bar_time = str(last.name)
    current_price = fetch_ticker_price(SYMBOL)

    # --- If we already checked this bar on a prior cycle, skip signal check ---
    same_bar = state.get("last_update_bar_time") == last_bar_time

    if not state["open"]:
        # Flat: look for entry
        entry_fired = False
        if not same_bar and check_entry_signal(df, macro_close, macro_ema):
            state["open"] = True
            state["entry_price"] = current_price
            state["entry_time"] = last_bar_time
            state["target_hit"] = False
            state["last_signal_time"] = last_bar_time
            alert_entry(cfg, current_price, last.name)
            entry_fired = True

        # Track-level pulse: send a 15-min market snapshot even when flat
        # (only for tracks with always_pulse=True). Skip if an entry alert
        # already fired this cycle to avoid double-notifying.
        if not entry_fired and cfg.get("always_pulse"):
            alert_market_pulse(cfg, current_price, float(last["ema_fast"]),
                               float(last["ema_slow"]), macro_close, macro_ema)
    else:
        entry_price = state["entry_price"]

        # 1) Stop check (Track A only)
        if cfg["stop_pct"] is not None:
            stop_price = entry_price * (1 - cfg["stop_pct"] / 100.0)
            if current_price <= stop_price:
                alert_stop_hit(cfg, entry_price, current_price)
                state = {"open": False, "entry_price": None, "entry_time": None,
                         "target_hit": False, "last_signal_time": last_bar_time,
                         "last_update_bar_time": last_bar_time}
                save_state(cfg["state_file"], state)
                return

        # 2) Opposite crossover — session ended
        if check_exit_crossover(df):
            alert_trend_end(cfg, entry_price, current_price)
            state = {"open": False, "entry_price": None, "entry_time": None,
                     "target_hit": False, "last_signal_time": last_bar_time,
                     "last_update_bar_time": last_bar_time}
            save_state(cfg["state_file"], state)
            return

        # 3) Target hit? Fire once, then keep tracking
        target_price = entry_price * (1 + TARGET_PCT / 100.0)
        just_hit_target = (not state["target_hit"]) and current_price >= target_price
        if just_hit_target:
            state["target_hit"] = True
            alert_target_hit(cfg, entry_price, current_price)

        # 4) Regular price update every 15 min
        alert_price_update(cfg, entry_price, current_price, state["target_hit"])

    state["last_update_bar_time"] = last_bar_time
    save_state(cfg["state_file"], state)


# =====================================================================
# Main loop
# =====================================================================

def sleep_until_next_slot():
    now = time.time()
    slot = POLL_INTERVAL_SECONDS
    next_slot_epoch = (int(now // slot) + 1) * slot + POLL_ALIGN_OFFSET_SECONDS
    wait = next_slot_epoch - now
    if wait > 0:
        print(f"Sleeping {int(wait)}s until next slot ...")
        time.sleep(wait)


def run_one_cycle():
    macro_close, macro_ema = get_macro_state(SYMBOL)
    print(f"\n=== Cycle {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} ===")
    print(f"Macro: 4h close={macro_close:.2f}, 4h 200EMA={macro_ema:.2f}, "
          f"macro_ok={macro_close > macro_ema}")
    for cfg in (TRACK_A, TRACK_B):
        try:
            process_track(cfg, macro_close, macro_ema)
        except Exception as e:
            print(f"[{cfg['short']}] error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="Run a single cycle then exit (for cron/GitHub Actions).")
    args = parser.parse_args()

    if args.once:
        # Single-shot mode: one poll then exit. Used by GitHub Actions cron.
        try:
            run_one_cycle()
        except Exception as e:
            print(f"cycle error: {e}")
            try:
                send_telegram(f"⚠️ Alerter cycle error: {e}")
            except Exception:
                pass
            sys.exit(1)
        return

    # Continuous mode: infinite loop for a VPS / local machine.
    print(f"Dual-track alerter starting for {SYMBOL} at {datetime.now(tz=timezone.utc)}")
    send_telegram(
        f"✅ Dual-track alerter started for {SYMBOL}\n"
        f"Track A: 1h 20/50, -5% stop, +{TARGET_PCT:.0f}% target\n"
        f"Track B: 15m 9/21, no stop, +{TARGET_PCT:.0f}% target\n"
        f"Price updates every 15 min while a track is open."
    )
    while True:
        try:
            run_one_cycle()
        except Exception as e:
            print(f"cycle error: {e}")
            try:
                send_telegram(f"⚠️ Alerter cycle error: {e}")
            except Exception:
                pass
        sleep_until_next_slot()


if __name__ == "__main__":
    main()
