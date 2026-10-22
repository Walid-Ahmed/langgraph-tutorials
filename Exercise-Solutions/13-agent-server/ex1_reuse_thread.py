# Exercise 16.1 — two PINGs on one thread, then GET_STATUS.
#
# This script creates a thread, submits PING twice, then GET_STATUS, and
# prints the response. It demonstrates that ping_count is per thread.
#
# Why this matters:
# - A new thread would report 0. Reusing the id is the whole exercise.
# - It does not edit the chapter's graph.
#
# Before you run this:
# 1. From 13-Agent-Server/pending-command-demo/:
#       langgraph dev --no-browser
# 2. Use that folder's virtual environment (it has langgraph-sdk).
#
# Run it with (from the repository root, server already up):
#    13-Agent-Server/pending-command-demo/.venv/bin/python Exercise-Solutions/13-agent-server/ex1_reuse_thread.py
#
# Expected:
#    {"status": "ok", "ping_count": 2}

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "13-Agent-Server",
        "pending-command-demo",
        "clients",
    ),
)

from common import run_command  # noqa: E402
from langgraph_sdk import get_sync_client  # noqa: E402

client = get_sync_client(url=os.getenv("SERVER_URL", "http://127.0.0.1:2024"))
thread_id = client.threads.create()["thread_id"]
run_command(thread_id, "PING")
run_command(thread_id, "PING")
state = run_command(thread_id, "GET_STATUS")
print(state["response"])
