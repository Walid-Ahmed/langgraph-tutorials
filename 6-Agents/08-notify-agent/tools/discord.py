# Tool: post a message to a Discord channel through a webhook.
#
# Part of 6-Agents/08-notify-agent (book Chapter 8, "An agent that acts").
# Not run directly; agent.py imports it.
#
# Setup (once): in Discord, open the channel's settings -> Integrations ->
# Webhooks -> New Webhook, copy the webhook URL, and put it in
# DISCORD_WEBHOOK_URL in the repo-root .env. No bot or account token needed.

import os

import requests
from langchain_core.tools import tool


@tool
def send_discord_message(text: str) -> str:
    """Post a message to the user's team Discord channel.

    Use this for updates meant for the team channel rather than the user alone.
    The channel is fixed by configuration; you cannot choose another one.
    """
    if os.getenv("NOTIFY_DRY_RUN") == "1":
        print(f"[DRY RUN] Discord -> {text!r}")
        return "Dry run: the Discord message was not really sent."

    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return "Discord is not configured (DISCORD_WEBHOOK_URL missing)."

    # Discord limits a message to 2000 characters; trim rather than fail.
    response = requests.post(webhook_url, json={"content": text[:2000]}, timeout=10)
    if response.ok:                       # Discord answers 204 No Content on success
        return "Discord message posted."
    return f"Discord error {response.status_code}: {response.text[:200]}"
