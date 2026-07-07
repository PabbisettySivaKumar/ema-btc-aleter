"""
Event-driven backtest engine.

Simulates:
 - Signal detection on closed candles (no lookahead)
 - Entry fill on the NEXT candle's open (per config.ENTRY_ON_NEXT_OPEN)
 - Position sizing from % risk per trade
 - Stop loss: swing level -> EMA21 -> 1.5x ATR (per doc's stated hierarchy)
 - TP1 (1R, close 50%) and TP2 (2R, close remainder) with EMA21 trailing
 - Exit on EMA9/21 cross-back or close below/above EMA21
 - Daily loss circuit breaker and max concurrent open positions
"""

import pandas as pd
import numpy as np

from strategy.signals import find_swings, evaluate_bar


class Trade:
    def __init__(self, direction, entry_time, entry_price, stop_price, size, risk_amount):
        self.direction = direction
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.initial_stop = stop_price
        self.size = size            # total position size (units of asset)
        self.remaining_size = size
        self.risk_amount = risk_amount  # $ risked (== size * |entry-stop|)
        r_distance = abs(entry_price - stop_price)
        if direction == "long":
            self.tp1_price = entry_price + r_distance * 1.0
            self.tp2_price = entry_price + r_distance * 2.0
        else:
            self.tp1_price = entry_price - r_distance * 1.0
            self.tp2_price = entry_price - r_distance * 2.0
        self.tp1_hit = False
        self.closed = False
        self.exit_time = None
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.r_multiple = 0.0
        self.exit_reason = None

    def unrealized_r_distance(self):
        return abs(self.entry_price - self.initial_stop)


class BacktestEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.equity = cfg.STARTING_EQUITY
        self.equity_curve = []
        self.open_positions = []
        self.closed_trades = []
        self.current_day = None
        self.day_start_equity = cfg.STARTING_EQUITY
        self.day_realized_pnl = 0.0

    def _stop_price_for(self, row, direction, structure_stop):
        cfg = self.cfg
        atr = row["atr"] if not np.isnan(row["atr"]) else 0
        min_distance = cfg.MIN_STOP_DISTANCE_ATR_MULT * atr

        if structure_stop is not None:
            # doc preference #1: recent swing low/high
            stop = structure_stop
        elif not np.isnan(row["ema_slow"]):
            # doc preference #2: EMA21
            stop = row["ema_slow"]
        else:
            # doc preference #3: 1.5x ATR
            stop = row["close"] - 1.5 * atr if direction == "long" else row["close"] + 1.5 * atr

        # Safety floor: right at an EMA9/21 crossover, EMA21 can sit almost
        # exactly at price, producing a near-zero stop distance and a wildly
        # oversized position. Never let the stop be tighter than a minimum
        # ATR-based distance from the reference close price.
        if direction == "long":
            max_allowed_stop = row["close"] - min_distance
            stop = min(stop, max_allowed_stop) if min_distance > 0 else stop
        else:
            min_allowed_stop = row["close"] + min_distance
            stop = max(stop, min_allowed_stop) if min_distance > 0 else stop

        return stop

    def _position_size(self, entry_price, stop_price):
        cfg = self.cfg
        distance = abs(entry_price - stop_price)
        if distance <= 0 or entry_price <= 0:
            return 0, 0

        # SPOT MODE: size as a fixed capital allocation (not derived from
        # stop distance), capped at no-leverage. Actual $ risk is whatever
        # falls out of (allocation x stop distance) - it is NOT forced to
        # hit RISK_PER_TRADE_PCT, since that would require leverage when
        # stops are tight.
        allocation = self.equity * (cfg.POSITION_ALLOCATION_PCT / 100.0)
        max_notional = self.equity * cfg.MAX_NOTIONAL_LEVERAGE
        notional = min(allocation, max_notional)
        size = notional / entry_price

        actual_risk = size * distance

        # Skip trades whose actual risk is too small relative to the
        # round-trip cost of taking them (entry + ~2 exit fills). These are
        # cost-dominated trades that can't clear a reasonable edge.
        est_fee = self._fill_cost(entry_price, size) * 3
        if actual_risk < cfg.MIN_RISK_TO_FEE_RATIO * est_fee:
            return 0, 0

        return size, actual_risk

    def _fill_cost(self, fill_price, size):
        """Fee + slippage cost in $ for one fill (entry, TP1 partial, or final exit)."""
        cost_pct = (self.cfg.TRADING_FEE_PCT + self.cfg.SLIPPAGE_PCT) / 100.0
        return fill_price * size * cost_pct

    def _reset_day_if_needed(self, ts):
        day = ts.normalize()
        if self.current_day is None or day != self.current_day:
            self.current_day = day
            self.day_start_equity = self.equity
            self.day_realized_pnl = 0.0

    def _daily_loss_exceeded(self):
        loss_pct = -self.day_realized_pnl / self.day_start_equity * 100.0
        return loss_pct >= self.cfg.MAX_DAILY_LOSS_PCT

    def _manage_open_positions(self, row, ts):
        still_open = []
        for pos in self.open_positions:
            exited = False

            tp1_on = getattr(self.cfg, "TP1_ENABLED", True)
            trail_ready = pos.tp1_hit or not tp1_on

            if pos.direction == "long":
                # Stop loss
                if row["low"] <= pos.stop_price:
                    pnl = (pos.stop_price - pos.entry_price) * pos.remaining_size
                    self._close_trade(pos, ts, pos.stop_price, pos.remaining_size, pnl, "stop_loss")
                    exited = True
                # TP1 (only if enabled)
                elif tp1_on and not pos.tp1_hit and row["high"] >= pos.tp1_price:
                    half = pos.remaining_size * 0.5
                    pnl = (pos.tp1_price - pos.entry_price) * half
                    fee = self._fill_cost(pos.tp1_price, half)
                    pos.remaining_size -= half
                    pos.tp1_hit = True
                    pos.stop_price = pos.entry_price  # move stop to breakeven after TP1
                    pos.realized_pnl += pnl - fee
                    pos.fees_paid += fee
                    self.equity += pnl - fee
                    self.day_realized_pnl += pnl - fee
                # TP2 / trail via EMA21 (active from entry when TP1 disabled)
                elif trail_ready and (row["high"] >= pos.tp2_price or row["close"] < row["ema_slow"]):
                    exit_price = pos.tp2_price if row["high"] >= pos.tp2_price else row["close"]
                    pnl = (exit_price - pos.entry_price) * pos.remaining_size
                    self._close_trade(pos, ts, exit_price, pos.remaining_size, pnl, "tp2_or_trail")
                    exited = True
                # EMA cross-back exit (full, if TP1 not yet hit)
                elif not pos.tp1_hit and row["ema_fast"] < row["ema_slow"]:
                    pnl = (row["close"] - pos.entry_price) * pos.remaining_size
                    self._close_trade(pos, ts, row["close"], pos.remaining_size, pnl, "ema_cross_exit")
                    exited = True

            else:  # short
                if row["high"] >= pos.stop_price:
                    pnl = (pos.entry_price - pos.stop_price) * pos.remaining_size
                    self._close_trade(pos, ts, pos.stop_price, pos.remaining_size, pnl, "stop_loss")
                    exited = True
                elif tp1_on and not pos.tp1_hit and row["low"] <= pos.tp1_price:
                    half = pos.remaining_size * 0.5
                    pnl = (pos.entry_price - pos.tp1_price) * half
                    fee = self._fill_cost(pos.tp1_price, half)
                    pos.remaining_size -= half
                    pos.tp1_hit = True
                    pos.stop_price = pos.entry_price
                    pos.realized_pnl += pnl - fee
                    pos.fees_paid += fee
                    self.equity += pnl - fee
                    self.day_realized_pnl += pnl - fee
                elif trail_ready and (row["low"] <= pos.tp2_price or row["close"] > row["ema_slow"]):
                    exit_price = pos.tp2_price if row["low"] <= pos.tp2_price else row["close"]
                    pnl = (pos.entry_price - exit_price) * pos.remaining_size
                    self._close_trade(pos, ts, exit_price, pos.remaining_size, pnl, "tp2_or_trail")
                    exited = True
                elif not pos.tp1_hit and row["ema_fast"] > row["ema_slow"]:
                    pnl = (pos.entry_price - row["close"]) * pos.remaining_size
                    self._close_trade(pos, ts, row["close"], pos.remaining_size, pnl, "ema_cross_exit")
                    exited = True

            if not exited:
                still_open.append(pos)

        self.open_positions = still_open

    def _close_trade(self, pos, ts, exit_price, exit_size, pnl, reason):
        fee = self._fill_cost(exit_price, exit_size)
        pos.closed = True
        pos.exit_time = ts
        pos.realized_pnl += pnl - fee  # accumulate on top of any TP1 partial already booked
        pos.fees_paid += fee
        pos.r_multiple = pos.realized_pnl / pos.risk_amount if pos.risk_amount else 0
        pos.exit_reason = reason
        self.equity += pnl - fee
        self.day_realized_pnl += pnl - fee
        self.closed_trades.append(pos)

    def run(self, df: pd.DataFrame) -> dict:
        cfg = self.cfg
        is_swing_high, is_swing_low = find_swings(df, cfg.SWING_LOOKBACK)

        pending_signal = None  # signal confirmed on bar i, to be filled at open of bar i+1

        for i in range(len(df)):
            row = df.iloc[i]
            ts = df.index[i]
            self._reset_day_if_needed(ts)

            # 1) Manage existing open positions using this bar's OHLC
            self._manage_open_positions(row, ts)

            # 2) Fill any pending signal at this bar's open
            if pending_signal is not None:
                direction, stop_price = pending_signal
                entry_price = row["open"]
                size, risk_amount = self._position_size(entry_price, stop_price)
                if size > 0 and len(self.open_positions) < cfg.MAX_OPEN_POSITIONS:
                    trade = Trade(direction, ts, entry_price, stop_price, size, risk_amount)
                    entry_fee = self._fill_cost(entry_price, size)
                    trade.fees_paid += entry_fee
                    trade.realized_pnl -= entry_fee
                    self.equity -= entry_fee
                    self.day_realized_pnl -= entry_fee
                    self.open_positions.append(trade)
                pending_signal = None

            # 3) Evaluate for a NEW signal on this (closed) bar
            if not self._daily_loss_exceeded() and len(self.open_positions) < cfg.MAX_OPEN_POSITIONS:
                result = evaluate_bar(df, is_swing_high, is_swing_low, i, cfg)
                if result and result["direction"] and result["score"] >= cfg.SCORE_THRESHOLD:
                    stop_price = self._stop_price_for(row, result["direction"], result["stop_level"])
                    pending_signal = (result["direction"], stop_price)

            self.equity_curve.append({"time": ts, "equity": self.equity})

        return self._summarize()

    def _summarize(self) -> dict:
        trades = self.closed_trades
        n = len(trades)
        equity_df = pd.DataFrame(self.equity_curve).set_index("time") if self.equity_curve else pd.DataFrame()

        if n == 0:
            return {
                "num_trades": 0,
                "win_rate_pct": 0.0,
                "total_return_pct": 0.0,
                "profit_factor": 0.0,
                "avg_r_multiple": 0.0,
                "max_drawdown_pct": 0.0,
                "final_equity": self.equity,
                "total_fees_paid": 0.0,
                "trades": [],
                "equity_curve": equity_df,
            }

        wins = [t for t in trades if t.realized_pnl > 0]
        losses = [t for t in trades if t.realized_pnl <= 0]
        gross_profit = sum(t.realized_pnl for t in wins)
        gross_loss = -sum(t.realized_pnl for t in losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

        equity_series = equity_df["equity"] if not equity_df.empty else pd.Series([self.cfg.STARTING_EQUITY])
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100.0
        max_dd = drawdown.min()

        return {
            "num_trades": n,
            "win_rate_pct": len(wins) / n * 100.0,
            "total_return_pct": (self.equity - self.cfg.STARTING_EQUITY) / self.cfg.STARTING_EQUITY * 100.0,
            "profit_factor": profit_factor,
            "avg_r_multiple": np.mean([t.r_multiple for t in trades]),
            "max_drawdown_pct": max_dd,
            "total_fees_paid": sum(t.fees_paid for t in trades),
            "final_equity": self.equity,
            "trades": trades,
            "equity_curve": equity_df,
        }
