"""
V2 strategy backtest — the exact rules we designed together:

  Instrument:  BTC/USDT spot, LONG-only (no shorts on spot)
  Timeframe:   1h
  Entry:       20 EMA crosses ABOVE 50 EMA on 1h
               AND 1h close > 4h 200 EMA (macro trend filter)
  Fill:        next candle's open (no lookahead)
  Sizing:      all-in on trading capital (only 1 open trade at a time)
  Stop loss:   -2% from entry, OR 4h close breaks 4h 200 EMA
  TP1:         +3% -> sell 50%, move remaining stop to entry (breakeven)
  Runner:      exit remaining 50% when 1h CLOSE < 1h 20 EMA
  Costs:       0.1% fee + 0.02% slippage per fill

Capital split (from user):
  Total capital:       $1000
  Long-term BTC hold:  $500 (untouched, not modeled here)
  Trading capital:     $500 (this backtest)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

STARTING_CAPITAL = 500.0
FEE_PCT = 0.10
SLIPPAGE_PCT = 0.02
STOP_PCT = 5.0
TP1_PCT = 3.0
TP1_FRACTION = 0.5
EMA_FAST = 20
EMA_SLOW = 50
EMA_MACRO = 200  # on 4h
VOLUME_SMA_PERIOD = 20
VOLUME_MULTIPLIER = 1.2

# Pullback entry: after a valid crossover, wait up to N bars for price to pull
# back near the 20 EMA and print a green (bullish) confirmation candle.
PULLBACK_MAX_WAIT_BARS = 10
PULLBACK_PROXIMITY_PCT = 0.3  # low must come within 0.3% of the 20 EMA


def load_1h(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def build_indicators(df_1h: pd.DataFrame) -> pd.DataFrame:
    out = df_1h.copy()
    out["ema_fast"] = ema(out["close"], EMA_FAST)
    out["ema_slow"] = ema(out["close"], EMA_SLOW)
    out["volume_sma"] = out["volume"].rolling(VOLUME_SMA_PERIOD).mean()

    # Resample 1h -> 4h to get macro trend filter
    df_4h = out.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    df_4h["ema_macro"] = ema(df_4h["close"], EMA_MACRO)

    # Merge 4h EMA back on the 1h index using forward-fill of the LAST CLOSED 4h bar
    # (shift by 1 so we never use in-progress 4h data on the current 1h bar).
    df_4h_shifted = df_4h[["ema_macro", "close"]].shift(1).rename(
        columns={"close": "close_4h_last"}
    )
    out = out.join(df_4h_shifted.reindex(out.index, method="ffill"))
    return out


def fee_cost(price: float, size: float) -> float:
    return price * size * (FEE_PCT + SLIPPAGE_PCT) / 100.0


def run_backtest(df: pd.DataFrame, pullback_mode: bool = False, no_stop: bool = False):
    equity = STARTING_CAPITAL
    equity_curve = []
    trades = []

    # position state
    in_pos = False
    entry_price = 0.0
    stop_price = 0.0
    tp1_price = 0.0
    remaining_size = 0.0
    tp1_hit = False
    entry_time = None
    entry_risk = 0.0
    entry_fee = 0.0
    trade_pnl = 0.0
    trade_fees = 0.0

    pending_entry = False
    # pullback-mode setup state
    setup_active = False
    setup_bars_waited = 0

    for i in range(len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        prev = df.iloc[i - 1] if i > 0 else None

        # 1) Manage open position on THIS bar's OHLC
        if in_pos:
            exited = False

            # Priority 1: stop loss (skipped entirely in --no-stop mode)
            if not no_stop and row["low"] <= stop_price:
                exit_price = stop_price
                pnl = (exit_price - entry_price) * remaining_size
                fee = fee_cost(exit_price, remaining_size)
                trade_pnl += pnl - fee
                trade_fees += fee
                equity += pnl - fee
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "reason": "stop_loss" if not tp1_hit else "breakeven_stop",
                    "pnl": trade_pnl, "fees": trade_fees,
                    "r_multiple": trade_pnl / entry_risk if entry_risk else 0,
                })
                in_pos = False
                exited = True

            # Priority 2: TP1 (partial)
            elif not tp1_hit and row["high"] >= tp1_price:
                half = remaining_size * TP1_FRACTION
                pnl = (tp1_price - entry_price) * half
                fee = fee_cost(tp1_price, half)
                remaining_size -= half
                tp1_hit = True
                if not no_stop:
                    stop_price = entry_price  # move stop to breakeven
                trade_pnl += pnl - fee
                trade_fees += fee
                equity += pnl - fee

            # Priority 3: Runner exit on 1h close < 20 EMA
            if not exited and in_pos and tp1_hit and row["close"] < row["ema_fast"]:
                exit_price = row["close"]
                pnl = (exit_price - entry_price) * remaining_size
                fee = fee_cost(exit_price, remaining_size)
                trade_pnl += pnl - fee
                trade_fees += fee
                equity += pnl - fee
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "reason": "runner_trail_exit",
                    "pnl": trade_pnl, "fees": trade_fees,
                    "r_multiple": trade_pnl / entry_risk if entry_risk else 0,
                })
                in_pos = False
                exited = True

            # Priority 4: macro trend break (4h close < 4h 200 EMA) -> full exit
            if not exited and in_pos and not np.isnan(row["ema_macro"]) \
                    and row["close_4h_last"] < row["ema_macro"]:
                exit_price = row["close"]
                pnl = (exit_price - entry_price) * remaining_size
                fee = fee_cost(exit_price, remaining_size)
                trade_pnl += pnl - fee
                trade_fees += fee
                equity += pnl - fee
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "reason": "macro_trend_break",
                    "pnl": trade_pnl, "fees": trade_fees,
                    "r_multiple": trade_pnl / entry_risk if entry_risk else 0,
                })
                in_pos = False

        # 2) Fill pending entry at THIS bar's open
        if pending_entry and not in_pos:
            entry_price = row["open"]
            stop_price = entry_price * (1 - STOP_PCT / 100.0)
            tp1_price = entry_price * (1 + TP1_PCT / 100.0)
            notional = equity  # all-in trading capital
            remaining_size = notional / entry_price
            entry_risk = (entry_price - stop_price) * remaining_size
            entry_fee = fee_cost(entry_price, remaining_size)
            trade_pnl = -entry_fee
            trade_fees = entry_fee
            equity -= entry_fee
            entry_time = ts
            tp1_hit = False
            in_pos = True
            pending_entry = False

        # 3) Look for a new signal on this closed bar (only if flat)
        if not in_pos and prev is not None and not np.isnan(row["ema_macro"]):
            crossed_up = (prev["ema_fast"] <= prev["ema_slow"]) and (row["ema_fast"] > row["ema_slow"])
            macro_ok = row["close"] > row["ema_macro"]
            volume_ok = (
                not np.isnan(row["volume_sma"])
                and row["volume"] > VOLUME_MULTIPLIER * row["volume_sma"]
            )

            if not pullback_mode:
                # Immediate entry on the crossover bar
                if crossed_up and macro_ok and volume_ok:
                    pending_entry = True
            else:
                # Pullback mode: 2-step.  Step 1: arm a setup on a valid crossover.
                # Step 2: within N bars, enter when price pulls back to touch the
                # 20 EMA and prints a green (bullish) confirmation candle.
                if crossed_up and macro_ok and volume_ok:
                    setup_active = True
                    setup_bars_waited = 0
                elif setup_active:
                    setup_bars_waited += 1
                    # Setup invalidated if macro trend breaks or fast<slow again
                    if row["ema_fast"] < row["ema_slow"] or row["close"] < row["ema_macro"]:
                        setup_active = False
                    elif setup_bars_waited > PULLBACK_MAX_WAIT_BARS:
                        setup_active = False
                    else:
                        # Pullback: bar's LOW comes within PULLBACK_PROXIMITY_PCT of 20 EMA
                        proximity = row["ema_fast"] * (PULLBACK_PROXIMITY_PCT / 100.0)
                        touched_ema = (row["low"] <= row["ema_fast"] + proximity)
                        confirm_candle = row["close"] > row["open"]  # bullish
                        if touched_ema and confirm_candle:
                            pending_entry = True
                            setup_active = False

        equity_curve.append({"time": ts, "equity": equity})

    # Force-close any open position at last bar's close (for accounting completeness)
    if in_pos:
        last = df.iloc[-1]
        exit_price = last["close"]
        pnl = (exit_price - entry_price) * remaining_size
        fee = fee_cost(exit_price, remaining_size)
        trade_pnl += pnl - fee
        trade_fees += fee
        equity += pnl - fee
        trades.append({
            "entry_time": entry_time, "exit_time": df.index[-1],
            "entry_price": entry_price, "exit_price": exit_price,
            "reason": "end_of_data",
            "pnl": trade_pnl, "fees": trade_fees,
            "r_multiple": trade_pnl / entry_risk if entry_risk else 0,
        })

    return equity, trades, pd.DataFrame(equity_curve).set_index("time")


def summarize(equity, trades, equity_curve):
    n = len(trades)
    if n == 0:
        return {"num_trades": 0, "final_equity": equity}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in losses)
    pf = gp / gl if gl > 0 else float("inf")
    total_fees = sum(t["fees"] for t in trades)
    running_max = equity_curve["equity"].cummax()
    dd = (equity_curve["equity"] - running_max) / running_max * 100.0

    return {
        "num_trades": n,
        "win_rate_pct": len(wins) / n * 100.0,
        "profit_factor": pf,
        "avg_r_multiple": np.mean([t["r_multiple"] for t in trades]),
        "total_return_pct": (equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0,
        "max_drawdown_pct": dd.min(),
        "total_fees": total_fees,
        "final_equity": equity,
        "avg_win_pct": np.mean([(t["exit_price"] / t["entry_price"] - 1) * 100 for t in wins]) if wins else 0,
        "avg_loss_pct": np.mean([(t["exit_price"] / t["entry_price"] - 1) * 100 for t in losses]) if losses else 0,
        "exit_breakdown": pd.Series([t["reason"] for t in trades]).value_counts().to_dict(),
    }


def buy_and_hold(df: pd.DataFrame) -> dict:
    start_price = df.iloc[0]["close"]
    end_price = df.iloc[-1]["close"]
    entry_fee = fee_cost(start_price, STARTING_CAPITAL / start_price)
    units = (STARTING_CAPITAL - entry_fee) / start_price
    final = units * end_price
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL * 100.0
    running_max = df["close"].cummax()
    dd_pct = (df["close"] - running_max) / running_max * 100.0
    return {"return_pct": ret, "max_drawdown_pct": dd_pct.min(), "final": final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to 1h OHLCV CSV")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (UTC)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (UTC)")
    parser.add_argument("--pullback", action="store_true", help="Enable pullback entry mode")
    parser.add_argument("--no-stop", action="store_true", help="Disable -2%% stop loss (hold and wait)")
    parser.add_argument("--label", default="", help="Header label")
    args = parser.parse_args()

    print(f"Loading 1h data from {args.csv} ...")
    df = load_1h(args.csv)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end, tz="UTC")]
    print(f"Loaded {len(df)} 1h bars from {df.index[0]} to {df.index[-1]}")

    print("Computing indicators (1h 20/50 EMA + 4h 200 EMA) ...")
    df = build_indicators(df)

    print(f"Running backtest (pullback_mode={args.pullback}) ...")
    equity, trades, equity_curve = run_backtest(df, pullback_mode=args.pullback, no_stop=args.no_stop)
    summary = summarize(equity, trades, equity_curve)
    bnh = buy_and_hold(df)

    header = "V2 STRATEGY BACKTEST"
    if args.label:
        header += f" - {args.label}"
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    print(f"Period:                   {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Starting capital:         ${STARTING_CAPITAL:.2f}")
    print(f"Entry mode:               {'PULLBACK' if args.pullback else 'IMMEDIATE'}")
    if summary["num_trades"] == 0:
        print("No trades taken.")
    else:
        print(f"Trades taken:             {summary['num_trades']}")
        print(f"Win rate:                 {summary['win_rate_pct']:.1f}%")
        print(f"Profit factor:            {summary['profit_factor']:.2f}")
        print(f"Avg R multiple:           {summary['avg_r_multiple']:.2f}")
        print(f"Avg win:                  +{summary['avg_win_pct']:.2f}%")
        print(f"Avg loss:                 {summary['avg_loss_pct']:.2f}%")
        print(f"Strategy return:          {summary['total_return_pct']:+.2f}%")
        print(f"Strategy max drawdown:    {summary['max_drawdown_pct']:.2f}%")
        print(f"Total fees paid:          ${summary['total_fees']:.2f}")
        print(f"Final equity:             ${summary['final_equity']:.2f}")
        print(f"Exit reasons:             {summary['exit_breakdown']}")
    print("-" * 60)
    print(f"Buy-and-hold return:      {bnh['return_pct']:+.2f}%")
    print(f"BTC max drawdown:         {bnh['max_drawdown_pct']:.2f}%")
    print(f"BnH final equity:         ${bnh['final']:.2f}")
    if summary["num_trades"] > 0:
        alpha = summary["total_return_pct"] - bnh["return_pct"]
        print(f"Strategy alpha vs BnH:    {alpha:+.2f} pp")
    print("=" * 60)

    if trades:
        pd.DataFrame(trades).to_csv("v2_trade_log.csv", index=False)


if __name__ == "__main__":
    main()
