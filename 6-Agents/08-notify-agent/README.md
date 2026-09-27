# 08 — An Agent That Acts: Telegram, Gmail and Discord

A `create_agent` agent with three tools that do something real: send you a Telegram message, send an email from your Gmail account, and post to a Discord channel. Book: Chapter 8, "An agent that acts".

```text
08-notify-agent/
├── agent.py            # builds and runs the agent
├── tools/
│   ├── __init__.py     # NOTIFY_TOOLS = [send_telegram_message, send_email, send_discord_message]
│   ├── telegram.py     # send_telegram_message(text)
│   ├── gmail.py        # send_email(to, subject, body)
│   └── discord.py      # send_discord_message(text)
└── .env.example        # the settings you need
```

## Run it

```bash
# 1. No accounts needed: prints what it would send
NOTIFY_DRY_RUN=1 python "6-Agents/08-notify-agent/agent.py"

# 2. For real, after filling in .env (see .env.example)
python "6-Agents/08-notify-agent/agent.py"
```

## Guardrails (all in code, none in the prompt)

| Guard | Where | Effect |
|---|---|---|
| Email allow-list | `tools/gmail.py` | only `ALLOWED_EMAIL_RECIPIENTS` (default: your own address) can receive email |
| Fixed Telegram chat and Discord channel | `tools/telegram.py`, `tools/discord.py` | the model cannot choose another recipient or channel |
| Dry run | all three tools | `NOTIFY_DRY_RUN=1` prints instead of sending |
| One email per run | `agent.py` | `ToolCallLimitMiddleware(tool_name="send_email", run_limit=1)` |
| Secrets stay out of the model | all three tools | tokens and passwords are read from `.env` inside the tool |
