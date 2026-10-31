# Exercise 16.5 — break it on purpose: compile the graph with no checkpointer.
#
# The service still passes a thread_id, and nothing raises. With no
# checkpointer there is nowhere to save the thread, so every request starts
# from empty state and the count never climbs.
#
# Why this matters:
# - On LangGraph Server you compile without a checkpointer because the server
#   stores threads. In your own FastAPI process nobody does, unless you attach one.
# - The failure is silent: 204s and 200s, with the wrong number.
#
# Before you run this:
#    python -m pip install -e ".[backend]"
#    from 13-Agent-Server/pending-command-demo/
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    python ../../Exercise-Solutions/13-agent-server/ex5_no_checkpointer.py
#
# Expected:
#    ping 204
#    ping 204
#    {'status': 'ok', 'ping_count': 0}

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app import app
from server.graph import builder


@asynccontextmanager
async def lifespan_without_checkpointer(app: FastAPI):
    """The broken lifespan: the same graph, compiled with nothing to save to."""
    app.state.graph = builder.compile()
    yield


# Swap only the lifespan, so the routes and auth are the chapter's own.
app.router.lifespan_context = lifespan_without_checkpointer

AMIRA = {"Authorization": "Bearer token-amira"}

with TestClient(app) as client:
    print("ping", client.post("/api/ping", headers=AMIRA).status_code)
    print("ping", client.post("/api/ping", headers=AMIRA).status_code)
    print(client.get("/api/status", headers=AMIRA).json())
