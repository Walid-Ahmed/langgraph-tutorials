# Run from the repository root:
#   python "6-Agents/08-notify-agent/agent.py"
#
# Try it without any accounts first (prints instead of sending):
#   NOTIFY_DRY_RUN=1 python "6-Agents/08-notify-agent/agent.py"
#
# What it does:
#   Builds a create_agent agent with three tools that act in the real world —
#   send_telegram_message (tools/telegram.py), send_email (tools/gmail.py) and
#   send_discord_message (tools/discord.py) — and gives it three requests:
#   a Telegram reminder, an email summary, and a team update on Discord.
#
# What it demonstrates (Chapter 8 — Agents, "An agent that acts"):
#   - tools kept in separate files and collected in one list (tools/__init__.py)
#   - secrets read from .env inside the tools, never passed through the model
#   - guardrails in CODE: an email allow-list, a dry-run switch, and a
#     ToolCallLimitMiddleware so each run sends at most one email
#   - failures returned to the model as text, so it can explain them
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env, plus (unless NOTIFY_DRY_RUN=1)
#   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GMAIL_ADDRESS, GMAIL_APP_PASSWORD,
#   DISCORD_WEBHOOK_URL.
#   See .env.example in this folder.
#
# Expected output (wording varies):
#   Request 1 -> tools used: ['send_telegram_message'] ; your phone gets the reminder
#   Request 2 -> tools used: ['send_email'] ; an email arrives in your own inbox
#   Request 3 -> tools used: ['send_discord_message'] ; the update appears in your channel

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")
sys.path.append(str(HERE))                 # so `import tools` finds ./tools
from tools import NOTIFY_TOOLS  # noqa: E402

MY_EMAIL = os.getenv("GMAIL_ADDRESS", "me@example.com")

SYSTEM_PROMPT = (
    "You are a personal assistant that can notify the user. "
    "Use send_telegram_message for short reminders. "
    f"Use send_email for longer content; the user's own address is {MY_EMAIL}. "
    "Use send_discord_message for updates meant for the team channel. "
    "Send each message once, then confirm in one sentence what you sent."
)

REQUESTS = [
    "Send me a Telegram reminder to review Chapter 8 tonight at 9.",
    "Email me a three-line summary of what an AI agent is, subject 'Agent notes'.",
    "Tell the team on Discord that the Chapter 8 draft is ready for review.",
]


def build_agent():
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=NOTIFY_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        # At most one email per run, whatever the model decides.
        middleware=[ToolCallLimitMiddleware(tool_name="send_email", run_limit=1)],
    )


def main() -> None:
    agent = build_agent()
    for request in REQUESTS:
        result = agent.invoke({"messages": [HumanMessage(request)]})
        used = [(m.name, m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
        print(f"\nRequest: {request}")
        for name, outcome in used:
            print(f"  tool: {name} -> {outcome}")
        print(f"  agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
