"""
Notification senders for LIVE signal alerts (Telegram + WhatsApp).

NOT used during backtesting. This is here so the same signal-scoring code
(strategy/signals.py) can be reused later for live alert-only mode, once
you're ready to move past backtesting. No orders are ever placed from here.

--- Telegram setup ---
1. Message @BotFather on Telegram -> /newbot -> get a bot token.
2. Message your new bot once, then visit:
   https://api.telegram.org/bot<TOKEN>/getUpdates
   to find your chat_id.
3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID below (or as env vars).

--- WhatsApp setup ---
WhatsApp requires either:
  a) Twilio's WhatsApp Sandbox/Business API (easiest to start), or
  b) Meta's official WhatsApp Business Cloud API.
This stub uses Twilio's API as the default path. You'll need a Twilio
account SID, auth token, and an approved WhatsApp sender number.
"""

import os
import requests

# Load .env file from project root if present (so you don't need to
# `export` variables every session). Requires: pip install python-dotenv
try:
    from dotenv import load_dotenv
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass  # dotenv not installed - falls back to plain environment variables

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "")  # e.g. "whatsapp:+14155238886"
TWILIO_WHATSAPP_TO = os.environ.get("TWILIO_WHATSAPP_TO", "")      # your number, "whatsapp:+91..."


def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Not configured - skipping. Message was:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    resp.raise_for_status()


def send_whatsapp(message: str):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM, TWILIO_WHATSAPP_TO]):
        print("[WhatsApp] Not configured - skipping. Message was:\n", message)
        return
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    resp = requests.post(
        url,
        data={"From": TWILIO_WHATSAPP_FROM, "To": TWILIO_WHATSAPP_TO, "Body": message},
        auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
    )
    resp.raise_for_status()


def format_signal_message(symbol, direction, score, entry_price, stop_price, tp1, tp2, ts):
    arrow = "🟢 BUY" if direction == "long" else "🔴 SELL"
    return (
        f"{arrow} SIGNAL - {symbol}\n"
        f"Time: {ts}\n"
        f"Score: {score}/100\n"
        f"Suggested entry: {entry_price:.2f}\n"
        f"Stop loss: {stop_price:.2f}\n"
        f"TP1 (1R): {tp1:.2f}\n"
        f"TP2 (2R): {tp2:.2f}\n"
        f"⚠️ Review before placing - not auto-executed."
    )


def notify_all(message: str):
    send_telegram(message)
    send_whatsapp(message)
