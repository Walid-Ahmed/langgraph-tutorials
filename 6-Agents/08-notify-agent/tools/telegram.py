# Tool: send a Telegram message to YOUR chat through a Telegram bot.
#
# Part of 6-Agents/08-notify-agent (book Chapter 8, "An agent that acts").
# Not run directly; agent.py imports it.
#
# Setup (once): talk to @BotFather in Telegram, create a bot, copy its token
# into TELEGRAM_BOT_TOKEN; send your bot any message, then open
# https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id": ...}
# into TELEGRAM_CHAT_ID. Both go in the repo-root .env.

import os

import requests
from langchain_core.tools import tool


@tool
def send_telegram_message(text: str) -> str:
    """Send a short text message to the user's own Telegram chat.

    Use this for quick personal notifications and reminders. The message always
    goes to the user; you cannot choose another recipient.
    """
    if os.getenv("NOTIFY_DRY_RUN") == "1":
        # Dry run: show what WOULD be sent, send nothing.
        print(f"[DRY RUN] Telegram -> {text!r}")
        return "Dry run: the Telegram message was not really sent."

    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        # Returned, not raised: the model reads this and can tell the user.
        return "Telegram is not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing)."

    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    if response.ok:
        return "Telegram message sent."
    return f"Telegram error {response.status_code}: {response.text[:200]}"
