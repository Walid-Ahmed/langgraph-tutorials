# Exercise 16.3 — submit a command type the router does not know.
#
# This script creates a thread and streams a NOPE command. The router raises
# ValueError. TypedDict does not reject the JSON.
#
# Why this matters:
# - The failure shows up as an error event on the stream, not as a state update
#   that cleared the slot.
#
# Before you run this:
# 1. langgraph dev --no-browser
#    from 13-Agent-Server/pending-command-demo/
# 2. Use that folder's virtual environment.
#
# Run it with (from the repository root):
#    13-Agent-Server/pending-command-demo/.venv/bin/python Exercise-Solutions/13-agent-server/ex3_unknown_command.py
#
# Expected:
#    error {'error': 'ValueError', 'message': 'Unknown command type'}

import os

from langgraph_sdk import get_sync_client

client = get_sync_client(url=os.getenv("SERVER_URL", "http://127.0.0.1:2024"))
thread_id = client.threads.create()["thread_id"]
for chunk in client.runs.stream(
    thread_id,
    "commands",
    input={"pending_command": {"type": "NOPE"}, "response": None},
    stream_mode="values",
):
    print(chunk.event, chunk.data)
