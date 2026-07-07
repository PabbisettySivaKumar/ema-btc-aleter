"""
Signal generation: trend filter, EMA crossover, volume/RSI confirmation,
market structure (swing high/low break), pullback detection, and the
weighted trade score from the strategy doc.

IMPORTANT: every function here only looks at data up to and including the
current (closed) bar's index `i`. No forward-looking data is used, to keep
the backtest honest.
"""

import numpy as np
import pandas as pd


def find_swings(df: pd.DataFrame, lookback: int):
    """
    Fractal-style swing high/low detection.
    A bar is a swing high if its high is the max within +/- lookback bars.
    A bar is a swing low if its low is the min within +/- lookback bars.
    Returns two boolean Series: is_swing_high, is_swing_low.
    NOTE: by construction a swing point is only CONFIRMED `lookback` bars
    after it occurs (needs the right-side bars to exist) — the code below
    respects that when looking up "previous swing high/low" as of bar i.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max():
            is_swing_high[i] = True
        if lows[i] == window_l.min():
            is_swing_low[i] = True

    return pd.Series(is_swing_high, index=df.index), pd.Series(is_swing_low, index=df.index)


def last_confirmed_swing_level(df, is_swing_high, is_swing_low, i, lookback, direction):
    """
    Returns the most recent CONFIRMED swing high (direction='long') or
    swing low (direction='short') price known as of bar i (i.e. the swing
    bar plus `lookback` confirmation bars must be <= i).
    """
    confirm_cutoff = i - lookback
    if confirm_cutoff < 0:
        return None

    if direction == "long":
        candidates = np.where(is_swing_high.values[: confirm_cutoff + 1])[0]
        if len(candidates) == 0:
            return None
        idx = candidates[-1]
        return df["high"].iloc[idx], idx
    else:
        candidates = np.where(is_swing_low.values[: confirm_cutoff + 1])[0]
        if len(candidates) == 0:
            return None
        idx = candidates[-1]
        return df["low"].iloc[idx], idx


def evaluate_bar(df, is_swing_high, is_swing_low, i, cfg):
    """
    Evaluate all confirmation conditions for bar index i (a just-closed candle).
    Returns a dict with per-condition booleans, the direction ('long'/'short'/None),
    the total score, and supporting levels (swing stop level) if applicable.
    """
    if i < max(cfg.EMA_SLOW, cfg.VOLUME_SMA_PERIOD, cfg.RSI_PERIOD, cfg.SWING_LOOKBACK) + 1:
        return None  # not enough warmup data yet

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    result = {"direction": None, "score": 0, "conditions": {}, "stop_level": None}

    for direction in ("long", "short"):
        conditions = {}

        # --- Trend filter (HARD GATE - must align, per doc's "Trend Filter" section) ---
        if direction == "long":
            trend_ok = row["close"] > row["vwap"] and row["ema_fast"] > row["ema_slow"]
        else:
            trend_ok = row["close"] < row["vwap"] and row["ema_fast"] < row["ema_slow"]

        # --- EMA crossover (HARD GATE - the defining trigger event, must have just happened) ---
        if direction == "long":
            cross_ok = prev["ema_fast"] < prev["ema_slow"] and row["ema_fast"] > row["ema_slow"]
        else:
            cross_ok = prev["ema_fast"] > prev["ema_slow"] and row["ema_fast"] < row["ema_slow"]

        if not (trend_ok and cross_ok):
            # This direction doesn't even qualify for consideration - skip entirely.
            # (Previously this leaked a phantom score into `result` even when the
            # gate failed, which let the OTHER direction's threshold check be
            # bypassed. That was the bug causing far too many trades.)
            continue

        conditions["vwap"] = trend_ok
        conditions["ema_cross"] = cross_ok

        # --- Volume confirmation ---
        vol_ok = (
            not np.isnan(row["volume_sma"])
            and row["volume"] > cfg.VOLUME_MULTIPLIER * row["volume_sma"]
        )
        conditions["volume"] = vol_ok

        # --- RSI filter ---
        if direction == "long":
            rsi_ok = cfg.RSI_LONG_MIN <= row["rsi"] <= cfg.RSI_LONG_MAX and row["rsi"] < cfg.RSI_EXTREME_HIGH
        else:
            rsi_ok = cfg.RSI_SHORT_MIN <= row["rsi"] <= cfg.RSI_SHORT_MAX and row["rsi"] > cfg.RSI_EXTREME_LOW
        conditions["rsi"] = rsi_ok

        # --- Market structure: break of previous swing high/low ---
        swing = last_confirmed_swing_level(df, is_swing_high, is_swing_low, i, cfg.SWING_LOOKBACK, direction)
        structure_ok = False
        stop_level = None
        if swing is not None:
            level, _ = swing
            if direction == "long" and row["close"] > level:
                structure_ok = True
            elif direction == "short" and row["close"] < level:
                structure_ok = True
            stop_level = level
        conditions["structure"] = structure_ok

        # --- Pullback: recent pullback to EMA9 + confirmation candle ---
        # Look back a few bars for a touch/close near EMA9, then require current
        # candle to close in the trade direction (confirmation candle).
        pullback_ok = False
        lookback_window = df.iloc[max(0, i - 5): i + 1]
        near_ema9 = (
            (lookback_window["low"] <= lookback_window["ema_fast"] * 1.003)
            & (lookback_window["low"] >= lookback_window["ema_fast"] * 0.985)
        ).any() if direction == "long" else (
            (lookback_window["high"] >= lookback_window["ema_fast"] * 0.997)
            & (lookback_window["high"] <= lookback_window["ema_fast"] * 1.015)
        ).any()
        confirm_candle = row["close"] > row["open"] if direction == "long" else row["close"] < row["open"]
        pullback_ok = bool(near_ema9 and confirm_candle)
        conditions["pullback"] = pullback_ok

        # --- Resistance filter (documented assumption, see config.py) ---
        opp_swing = last_confirmed_swing_level(
            df, is_swing_high, is_swing_low, i, cfg.SWING_LOOKBACK,
            "long" if direction == "short" else "short"
        )
        resistance_ok = True
        if opp_swing is not None:
            level, _ = opp_swing
            buffer = level * (cfg.RESISTANCE_BUFFER_PCT / 100.0)
            if direction == "long" and level - buffer <= row["close"] <= level + buffer:
                resistance_ok = False
            if direction == "short" and level - buffer <= row["close"] <= level + buffer:
                resistance_ok = False
        conditions["resistance_filter"] = resistance_ok

        # --- Score ---
        score = sum(
            cfg.SCORE_WEIGHTS[k] for k, ok in conditions.items() if ok
        )

        if score > result["score"]:
            result["direction"] = direction
            result["score"] = score
            result["conditions"] = conditions
            result["stop_level"] = stop_level

    return result
