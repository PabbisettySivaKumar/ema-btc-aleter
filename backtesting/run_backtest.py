"""
Main entry point: loads OHLCV data, computes indicators, runs the backtest,
prints a performance report, and saves an equity curve chart.

Usage (with real data, after running data/fetch_binance_data.py):
    python backtesting/run_backtest.py --csv data/historical_BTCUSDT_15m.csv

Usage (with bundled synthetic sample data, just to sanity-check the code):
    python backtesting/run_backtest.py --csv data/sample_ohlcv_15m.csv --sample
"""

import argparse
import sys
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config as cfg
from indicators.indicators import compute_all_indicators
from backtesting.backtest_engine import BacktestEngine


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.sort_index()
    return df


def print_report(summary: dict, sample_warning: bool):
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    if sample_warning:
        print("!! Ran on SYNTHETIC sample data - NOT a real performance estimate !!\n")
    print(f"Trades taken:        {summary['num_trades']}")
    print(f"Win rate:            {summary['win_rate_pct']:.1f}%")
    print(f"Profit factor:       {summary['profit_factor']:.2f}")
    print(f"Avg R multiple:      {summary['avg_r_multiple']:.2f}")
    print(f"Total return:        {summary['total_return_pct']:.2f}%")
    print(f"Max drawdown:        {summary['max_drawdown_pct']:.2f}%")
    print(f"Total fees paid:     ${summary['total_fees_paid']:.2f}")
    print(f"Final equity:        ${summary['final_equity']:.2f}")
    print("=" * 50)


def save_equity_curve(summary: dict, out_path: str):
    eq = summary["equity_curve"]
    if eq.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(eq.index, eq["equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Equity curve chart saved -> {out_path}")


def save_trade_log(summary: dict, out_path: str):
    trades = summary["trades"]
    if not trades:
        return
    rows = []
    for t in trades:
        rows.append({
            "entry_time": t.entry_time, "direction": t.direction,
            "entry_price": t.entry_price, "exit_time": t.exit_time,
            "exit_reason": t.exit_reason, "realized_pnl": t.realized_pnl,
            "r_multiple": t.r_multiple,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Trade log saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--sample", action="store_true", help="Flag that this is synthetic sample data")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    print(f"Loading data from {args.csv} ...")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    print("Computing indicators ...")
    df = compute_all_indicators(df, cfg)

    print("Running backtest ...")
    engine = BacktestEngine(cfg)
    summary = engine.run(df)

    print_report(summary, sample_warning=args.sample)
    save_equity_curve(summary, os.path.join(args.out_dir, "equity_curve.png"))
    save_trade_log(summary, os.path.join(args.out_dir, "trade_log.csv"))


if __name__ == "__main__":
    main()
