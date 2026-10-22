# The one-node graph LangGraph Server runs for Chapter 16.
#
# A client writes a small command into pending_command. This router calls the
# matching function and returns that update with the slot set back to None.
#
# Why this matters:
# - One command is processed per run. The slot is not a queue.
# - Clearing happens in this return. If the handler raises, this return never
#   runs, so the command can still be in the saved state.
# - No checkpointer is attached. LangGraph Server stores the thread. Calling
#   graph.invoke yourself would forget the thread between calls (Chapter 9).
#
# Before you run this:
# 1. From 13-Agent-Server/pending-command-demo/, with Python 3.12:
#       python3.12 -m venv .venv
#       source .venv/bin/activate
#       python -m pip install -e .
# 2. No API key and no database.
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    langgraph dev --no-browser
#
# Then run:
#    python clients/ping.py

from langgraph.graph import END, START, StateGraph

from server.command_types import State
from server.handlers import handle_get_status, handle_ping

HANDLERS = {
    "PING": handle_ping,
    "GET_STATUS": handle_get_status,
}


def router(state: State) -> State:
    """Call the handler for pending_command, then clear the slot."""
    command = state.get("pending_command")
    if command is None:
        # An empty dict would leave a previous reply in place.
        return {"pending_command": None, "response": None}
    handler = HANDLERS.get(command.get("type"))
    if handler is None:
        raise ValueError("Unknown command type")
    update = handler(state)
    # The clear comes after the handler's keys, so a handler cannot keep the command.
    return {**update, "pending_command": None}


builder = StateGraph(State)
builder.add_node("router", router)
builder.add_edge(START, "router")
builder.add_edge("router", END)
# The server owns persistence. Do not pass a checkpointer here.
graph = builder.compile()
