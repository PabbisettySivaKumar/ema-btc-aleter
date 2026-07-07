"""
Indicator calculations: EMA, VWAP (daily-reset), RSI (Wilder), Volume SMA, ATR (Wilder).
All functions are vectorized with pandas and take/return a DataFrame with
columns: ['open','high','low','close','volume'] indexed by UTC timestamp.
"""

import numpy as np
import pandas as pd


def add_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def add_vwap_daily(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets every UTC day (standard for 24/7 crypto markets)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]

    day = df.index.normalize()
    cum_tp_vol = tp_vol.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when undefined (e.g. no losses yet)
    return rsi


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(window=period).mean()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def compute_all_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Adds all indicator columns needed by the strategy. Returns a new DataFrame."""
    out = df.copy()
    out["ema_fast"] = add_ema(out, cfg.EMA_FAST)
    out["ema_slow"] = add_ema(out, cfg.EMA_SLOW)
    out["vwap"] = add_vwap_daily(out)
    out["rsi"] = add_rsi(out, cfg.RSI_PERIOD)
    out["volume_sma"] = add_volume_sma(out, cfg.VOLUME_SMA_PERIOD)
    out["atr"] = add_atr(out, cfg.ATR_PERIOD)
    return out
