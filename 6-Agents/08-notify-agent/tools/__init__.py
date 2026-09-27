# The notify agent's toolbox. Each tool lives in its own file; this module only
# collects them, so agent.py can do `from tools import NOTIFY_TOOLS`.
from .discord import send_discord_message
from .gmail import send_email
from .telegram import send_telegram_message

NOTIFY_TOOLS = [send_telegram_message, send_email, send_discord_message]
