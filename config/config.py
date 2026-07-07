"""
Central configuration for the EMA9/21 + VWAP + Volume Confirmation Strategy.
Edit values here to tune the strategy without touching logic files.
"""

# --- Instrument / Data ---
SYMBOL = "BTCUSDT"          # Binance symbol
TIMEFRAME = "1h"           # 1-hour to reduce noise
EXCHANGE = "binance"

# --- Indicator periods ---
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
VOLUME_SMA_PERIOD = 20
ATR_PERIOD = 14

# --- Swing structure detection ---
SWING_LOOKBACK = 3          # bars on each side to confirm a swing high/low (fractal method)

# --- Filters (from strategy doc) ---
VOLUME_MULTIPLIER = 1.2     # volume must exceed 1.2x its 20-SMA
RSI_LONG_MIN, RSI_LONG_MAX = 55, 70
RSI_SHORT_MIN, RSI_SHORT_MAX = 30, 45
RSI_EXTREME_HIGH = 80        # avoid longs above this
RSI_EXTREME_LOW = 20         # avoid shorts below this

# --- Resistance filter (NOT explicitly defined in the source doc; documented assumption) ---
# We treat "resistance filter" as: no opposing swing level within this % of entry price
# in the direction of the trade (i.e. don't buy right into a nearby swing high).
RESISTANCE_BUFFER_PCT = 0.3   # percent

# --- Scoring weights (must sum to 100, per doc) ---
SCORE_WEIGHTS = {
    "ema_cross": 20,
    "vwap": 15,
    "volume": 15,
    "rsi": 10,
    "structure": 20,
    "pullback": 10,
    "resistance_filter": 10,
}
SCORE_THRESHOLD = 85

# --- Risk management ---
RISK_PER_TRADE_PCT = 1.0     # target risk %, used only for REPORTING/reference on spot
MAX_DAILY_LOSS_PCT = 3.0     # stop trading for the day after this much loss
MAX_OPEN_POSITIONS = 2
TP1_R = 1.0                  # take-profit 1 at 1R (close 50%)
TP2_R = 3.0                  # take-profit 2 at 3R (close remaining, or trail)
TP1_ENABLED = False          # False: skip TP1 half-close and let winners run to TP2/EMA21 trail
STARTING_EQUITY = 1000.0     # backtest starting capital (USD) - set to your real capital

# --- Trading costs ---
TRADING_FEE_PCT = 0.1        # Binance spot taker fee per fill (~0.1%). Check your
                              # actual tier - VIP tiers or maker orders can be lower.
SLIPPAGE_PCT = 0.02           # extra assumed slippage per fill, conservative default

# --- Position sizing: SPOT MODE (no leverage/margin) ---
# On spot, you can't size a position purely from "risk % / stop distance" the
# way a leveraged futures account can, because a tight stop would demand a
# position bigger than your account. Instead, size is a fixed SLICE of
# capital per trade. Actual risk % (based on the real stop distance) will
# vary trade-to-trade and is reported per-trade in the trade log, rather
# than targeted directly.
SPOT_MODE = True
POSITION_ALLOCATION_PCT = 100.0 / MAX_OPEN_POSITIONS  # equal split across max concurrent slots
MAX_NOTIONAL_LEVERAGE = 1.0    # hard safety ceiling - never exceed 1x equity, no leverage

# Skip a trade if its actual $ risk (stop distance x size) is too small
# relative to round-trip trading costs - such trades are cost-dominated and
# not worth taking regardless of the signal score.
MIN_RISK_TO_FEE_RATIO = 3.0

MIN_STOP_DISTANCE_ATR_MULT = 0.5  # stop must be at least this many ATRs away from entry

# --- Backtest execution assumption ---
# Signals are confirmed on a CLOSED candle; entry is simulated at the OPEN of the
# NEXT candle to avoid lookahead bias. This is a documented assumption since the
# source doc does not specify exact fill timing.
ENTRY_ON_NEXT_OPEN = True
