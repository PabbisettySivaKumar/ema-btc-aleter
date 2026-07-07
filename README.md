# EMA 9/21 + VWAP + Volume Confirmation Strategy — Backtest Engine

Implements the strategy from your documentation: EMA9/21 crossover, VWAP trend
filter, volume confirmation, RSI filter, market structure (swing break),
pullback entry, weighted scoring (≥80 to trade), and the stated risk rules
(1% risk/trade, 3% daily loss stop, max 2 open positions, TP1/TP2 with EMA21
trailing).

## ⚠️ Read this first
- A few strategy rules were ambiguous in the source doc and required
  documented assumptions (see `config/config.py` comments): VWAP reset
  period, exact "resistance filter" definition, and entry timing (next-bar
  open, to avoid lookahead bias). Review these and adjust if they don't
  match your intent.

## Setup (run these on your own machine, not in this sandbox)

```bash
cd ema_agent_project
pip install pandas numpy matplotlib requests

# 1. Get real historical data from Binance (public endpoint, no API key needed)
python data/fetch_binance_data.py --symbol BTCUSDT --interval 15m --days 180

# 2. Run the backtest on that real data
python backtesting/run_backtest.py --csv historical_BTCUSDT_15m.csv
```

This prints:
- Number of trades, win rate, profit factor, average R multiple
- Total return %, max drawdown %
- Saves `equity_curve.png` and `trade_log.csv` (every trade with entry/exit/P&L)

## Try more history / other symbols
```bash
python data/fetch_binance_data.py --symbol ETHUSDT --interval 15m --days 365
python backtesting/run_backtest.py --csv historical_ETHUSDT_15m.csv
```
Run it across several symbols and time ranges before trusting the numbers —
a strategy that only works on one coin/period is likely overfit.

## Project structure
```
config/         strategy parameters (weights, thresholds, risk %)
data/           data fetching (Binance)
indicators/     EMA, VWAP, RSI, Volume SMA, ATR
strategy/       signal logic, swing detection, scoring
backtesting/    backtest engine + main runner script
alerts/         Telegram + WhatsApp notifier (for LATER live alert-only mode)
```

## Next steps (after you're happy with backtest results)
1. Run on multiple symbols/timeframes/date ranges to check robustness.
2. **Live alert-only mode is ready** — see below.
3. Only after manually reviewing live alerts for a while would you
   consider connecting an execution layer — and even then, always with a
   manual confirm step, per what you described.

## Live alert-only monitor (Binance Spot, no leverage)

`live/live_monitor.py` watches the configured symbol/timeframe and sends you
a Telegram + WhatsApp message every time an EMA9/21 crossover happens, with:
- direction (long/short), confirmation score out of 100
- a clear **TAKE** or **SKIP** recommendation (score ≥ 80 AND the trade
  clears the same fee/risk sanity check used in backtesting)
- suggested entry, stop, TP1, TP2 for you to review and place manually

**It never places orders.** It only reads public market data and notifies you.

Setup:
```bash
pip install pandas numpy requests
# set up Telegram + WhatsApp per comments in alerts/notifier.py, then:
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
export TWILIO_WHATSAPP_FROM=...
export TWILIO_WHATSAPP_TO=...

python live/live_monitor.py
```
Leave it running (screen/tmux on a VPS, or a background service) — it polls
once per candle close (every 15 minutes by default). Before running, set
`STARTING_EQUITY` in `config/config.py` to your **real current capital** so
the position-sizing and cost-sanity check in the alert reflect your actual
account, not a placeholder number.

## Notifier setup (for later, not needed for backtesting)
See comments at the top of `alerts/notifier.py` for Telegram bot token /
chat ID setup, and Twilio WhatsApp sandbox setup.
