"""
SIMPLE CROSSOVER ALERT MONITOR (Binance Spot)

The simplest version, per your request:
  - EMA9 crosses ABOVE EMA21  ->  🟢 BUY alert to Telegram (+WhatsApp if configured)
  - EMA9 crosses BELOW EMA21  ->  🔴 SELL/EXIT alert

No filtering, no score, no take/skip verdict - every crossover gets an alert
and YOU decide what to do. This script never places orders.

NOTE for spot trading: a bearish (downward) cross can't be traded as a short
on spot - treat it as your EXIT signal for any position you're holding.

Run:
    python live/simple_crossover_alert.py
Polls once per candle close (15m by default, from config).
"""

import sys
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config as cfg
from indicators.indicators import add_ema
from alerts.notifier import notify_all

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


def fetch_recent_candles(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
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
    for c in ["open", "close"]:
        df[c] = df[c].astype(float)

    # Drop the currently-forming (unclosed) candle
    now = pd.Timestamp.now(tz="UTC")
    if df.iloc[-1]["close_time"] > now:
        df = df.iloc[:-1]

    return df.set_index("open_time")


def detect_crossover(df: pd.DataFrame):
    """
    Returns 'bullish', 'bearish', or None for the LATEST closed candle.
    """
    df = df.copy()
    df["ema_fast"] = add_ema(df, cfg.EMA_FAST, col="close")
    df["ema_slow"] = add_ema(df, cfg.EMA_SLOW, col="close")

    if len(df) < cfg.EMA_SLOW + 2:
        return None, None

    cur = df.iloc[-1]
    prev = df.iloc[-2]

    if prev["ema_fast"] < prev["ema_slow"] and cur["ema_fast"] > cur["ema_slow"]:
        return "bullish", cur
    if prev["ema_fast"] > prev["ema_slow"] and cur["ema_fast"] < cur["ema_slow"]:
        return "bearish", cur
    return None, None


def build_message(kind: str, row) -> str:
    price = row["close"]
    ts = row.name
    if kind == "bullish":
        return (
            f"🟢 BUY SIGNAL - {cfg.SYMBOL} ({cfg.TIMEFRAME})\n"
            f"EMA9 crossed ABOVE EMA21\n"
            f"Time: {ts}\n"
            f"Price: {price:.2f}\n"
            f"⚠️ Review the chart before placing any order."
        )
    else:
        return (
            f"🔴 SELL/EXIT SIGNAL - {cfg.SYMBOL} ({cfg.TIMEFRAME})\n"
            f"EMA9 crossed BELOW EMA21\n"
            f"Time: {ts}\n"
            f"Price: {price:.2f}\n"
            f"If you're holding a position, consider exiting.\n"
            f"⚠️ Review the chart before acting."
        )


def run_forever():
    print(f"Simple crossover monitor: {cfg.SYMBOL} {cfg.TIMEFRAME}")
    print("Alerts on EVERY EMA9/21 crossover. No orders are ever placed.\n")
    poll_seconds = INTERVAL_SECONDS[cfg.TIMEFRAME]
    last_alerted_candle = None

    while True:
        try:
            df = fetch_recent_candles(cfg.SYMBOL, cfg.TIMEFRAME)
            kind, row = detect_crossover(df)
            if kind is not None and row.name != last_alerted_candle:
                msg = build_message(kind, row)
                print(f"[{datetime.now(timezone.utc)}] {kind.upper()} crossover:\n{msg}\n")
                notify_all(msg)
                last_alerted_candle = row.name  # avoid duplicate alerts for same candle
            else:
                print(f"[{datetime.now(timezone.utc)}] No new crossover.")
        except Exception as e:
            print(f"[{datetime.now(timezone.utc)}] Error: {e}")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_forever()
