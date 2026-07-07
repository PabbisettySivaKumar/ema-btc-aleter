"""
LIVE ALERT-ONLY MONITOR (Binance Spot, no leverage)

Watches for EMA9/21 crossovers on the configured symbol/timeframe. Whenever
a crossover just happened (the actual trigger event per the strategy doc),
it sends you a Telegram + WhatsApp message telling you:
  - direction (long/short)
  - the confirmation score out of 100
  - a clear TAKE or SKIP recommendation (score >= threshold AND the trade
    clears the fee/risk sanity check used in backtesting)
  - suggested entry/stop/TP1/TP2 levels, for YOU to review and place
    manually on Binance

This script NEVER places orders. It only reads public market data and
sends notifications. You stay in full manual control.

Run continuously (foreground, screen/tmux, or as a background service):
    python live/live_monitor.py

It polls once per candle close (every 15 minutes for the default config).
"""

import sys
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # loads .env file into environment - must happen BEFORE
               # importing alerts.notifier, since that module reads
               # os.environ.get(...) at import time.

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config as cfg
from indicators.indicators import compute_all_indicators
from strategy.signals import find_swings, evaluate_bar
from backtesting.backtest_engine import BacktestEngine
from alerts.notifier import notify_all, format_signal_message

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


def fetch_recent_candles(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """Fetch the most recent `limit` candles and drop the currently-forming one."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    rows = resp.json()

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # Drop the last candle if it hasn't closed yet
    now = pd.Timestamp.now(tz="UTC")
    if df.iloc[-1]["close_time"] > now:
        df = df.iloc[:-1]

    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]
    return df


def check_for_signal(df: pd.DataFrame):
    """
    Runs indicators + signal evaluation on the latest CLOSED candle.
    Returns a dict describing the result, or None if no crossover happened
    on the latest bar (i.e. nothing worth alerting about at all).
    """
    df = compute_all_indicators(df, cfg)
    is_swing_high, is_swing_low = find_swings(df, cfg.SWING_LOOKBACK)

    i = len(df) - 1
    result = evaluate_bar(df, is_swing_high, is_swing_low, i, cfg)

    if result is None or result["direction"] is None:
        return None  # no crossover (with trend alignment) on this bar at all

    row = df.iloc[i]
    direction = result["direction"]
    score = result["score"]

    # Reuse the exact same spot position-sizing + cost-sanity logic as the
    # backtester, so the live recommendation matches what we validated.
    engine = BacktestEngine(cfg)
    engine.equity = cfg.STARTING_EQUITY  # set this to your real current capital
    stop_price = engine._stop_price_for(row, direction, result["stop_level"])
    size, actual_risk = engine._position_size(row["close"], stop_price)

    score_ok = score >= cfg.SCORE_THRESHOLD
    cost_ok = size > 0  # _position_size already returns 0 if cost-dominated
    take_trade = score_ok and cost_ok

    r_distance = abs(row["close"] - stop_price)
    tp1 = row["close"] + r_distance if direction == "long" else row["close"] - r_distance
    tp2 = row["close"] + 2 * r_distance if direction == "long" else row["close"] - 2 * r_distance

    return {
        "direction": direction,
        "score": score,
        "take_trade": take_trade,
        "score_ok": score_ok,
        "cost_ok": cost_ok,
        "entry": row["close"],
        "stop": stop_price,
        "tp1": tp1,
        "tp2": tp2,
        "timestamp": df.index[i],
    }


def build_message(sig: dict) -> str:
    base = format_signal_message(
        cfg.SYMBOL, sig["direction"], sig["score"],
        sig["entry"], sig["stop"], sig["tp1"], sig["tp2"], sig["timestamp"],
    )
    if sig["take_trade"]:
        verdict = "\n✅ RECOMMENDATION: Meets score + cost thresholds - worth reviewing."
    else:
        reasons = []
        if not sig["score_ok"]:
            reasons.append(f"score {sig['score']} < {cfg.SCORE_THRESHOLD} threshold")
        if not sig["cost_ok"]:
            reasons.append("stop too tight relative to trading fees (not cost-justified)")
        verdict = f"\n⚠️ RECOMMENDATION: SKIP - {', '.join(reasons)}."
    return base + verdict


def run_forever():
    print(f"Starting live monitor: {cfg.SYMBOL} {cfg.TIMEFRAME}")
    print("Alert-only. No orders will ever be placed by this script.")
    poll_seconds = INTERVAL_SECONDS[cfg.TIMEFRAME]

    while True:
        try:
            df = fetch_recent_candles(cfg.SYMBOL, cfg.TIMEFRAME, limit=300)
            sig = check_for_signal(df)
            if sig is not None:
                msg = build_message(sig)
                print(f"\n[{datetime.now(timezone.utc)}] Signal detected:\n{msg}\n")
                notify_all(msg)
            else:
                print(f"[{datetime.now(timezone.utc)}] No crossover on latest candle.")
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Error: {e}")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_forever()
