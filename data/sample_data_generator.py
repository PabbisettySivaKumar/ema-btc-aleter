"""
Generates SYNTHETIC 15m OHLCV data for testing the pipeline end-to-end
when real exchange data isn't available (e.g. in an offline sandbox).

THIS IS NOT REAL MARKET DATA. Do not draw any profit conclusions from
backtests run on this — it exists purely to prove the code runs correctly.
Use fetch_binance_data.py for real data before trusting any results.
"""

import numpy as np
import pandas as pd


def generate_sample_ohlcv(bars=6000, start_price=60000.0, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt_index = pd.date_range("2024-01-01", periods=bars, freq="15min", tz="UTC")

    # Random walk with mild trending regimes to give the strategy something to catch
    returns = rng.normal(loc=0.0000, scale=0.0018, size=bars)
    # inject a few trending regimes
    for start in range(0, bars, 400):
        length = min(150, bars - start)
        drift = rng.choice([-1, 1]) * rng.uniform(0.0004, 0.0009)
        returns[start:start + length] += drift

    close = start_price * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.normal(0, 0.0015, bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.0015, bars)))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.lognormal(mean=3.0, sigma=0.5, size=bars) * 10

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dt_index,
    )
    # ensure high/low consistency
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


if __name__ == "__main__":
    df = generate_sample_ohlcv()
    df.to_csv("sample_ohlcv_15m.csv")
    print(f"Generated {len(df)} sample bars -> sample_ohlcv_15m.csv")
