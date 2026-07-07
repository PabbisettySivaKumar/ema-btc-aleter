"""
Fetches REAL historical OHLCV candles from Binance's public REST API
(no API key required for historical klines).

Run this on YOUR machine (this sandbox has no internet access to Binance).

Usage:
    pip install requests pandas
    python fetch_binance_data.py --symbol BTCUSDT --interval 15m --days 180

Saves a CSV to data/historical_<symbol>_<interval>.csv with columns:
open_time, open, high, low, close, volume
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit=1000):
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        last_open_time = rows[-1][0]
        cur = last_open_time + INTERVAL_MS[interval]
        if len(rows) < limit:
            break
        time.sleep(0.3)  # be nice to the rate limit
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m", choices=list(INTERVAL_MS.keys()))
    parser.add_argument("--days", type=int, default=180, help="how many days of history to fetch")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print(f"Fetching {args.symbol} {args.interval} candles from {start.date()} to {end.date()} ...")
    rows = fetch_klines(args.symbol, args.interval, start_ms, end_ms)
    print(f"Fetched {len(rows)} candles.")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time")

    out_path = args.out or f"historical_{args.symbol}_{args.interval}.csv"
    df.to_csv(out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
