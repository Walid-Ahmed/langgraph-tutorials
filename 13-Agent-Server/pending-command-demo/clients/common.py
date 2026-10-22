# Shared client for Chapter 16. ping.py and get_status.py import it.
#
# The client talks only to LangGraph Server over HTTP. It never imports the
# graph or the handlers.
#
# Why this matters:
# - stream_mode "values" emits the state after the input is applied, and again
#   after the node finishes. The first snapshot still holds the command.
# - The helper keeps the last snapshot and refuses to return while the command
#   is still set. An HTTP 200 on its own does not mean the handler ran.
#
# Before you run a client:
# 1. The server from server/graph.py must already be up:
#       langgraph dev --no-browser
#    from 13-Agent-Server/pending-command-demo/
# 2. Optional: SERVER_URL if the server is not on port 2024.
#    Optional: THREAD_ID to continue a thread a previous client printed.
#
# Run a client with (from 13-Agent-Server/pending-command-demo/):
#    python clients/ping.py

import os

from langgraph_sdk import get_sync_client

client = get_sync_client(
    url=os.getenv("SERVER_URL", "http://127.0.0.1:2024")
)


def get_thread_id() -> str:
    """Reuse THREAD_ID when set. That id is the server's checkpoint name."""
    existing = os.getenv("THREAD_ID")
    if existing:
        return existing
    return client.threads.create()["thread_id"]


def run_command(thread_id: str, command_type: str) -> dict:
    """Submit one command and return the state after the run finishes."""
    final_state = None
    # The first values event is the input, still holding the command. Keep the last.
    for chunk in client.runs.stream(
        thread_id,
        "commands",  # the graph name in langgraph.json, sent as assistant_id
        input={
            "pending_command": {"type": command_type},
            "response": None,  # drop a reply left by an earlier run
        },
        stream_mode="values",
    ):
        if chunk.event == "error":
            raise RuntimeError(chunk.data)
        if chunk.event == "values":
            final_state = chunk.data
    if final_state is None:
        raise RuntimeError("No state received")
    if final_state.get("pending_command") is not None:
        raise RuntimeError("Command was not cleared")
    return final_state
