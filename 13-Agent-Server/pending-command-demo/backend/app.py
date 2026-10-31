# A FastAPI service that runs the pending-command graph in its own process.
#
# This is option B from Chapter 16 with no LangGraph Server at all. The graph
# is the same builder server/graph.py defines, compiled here with a
# checkpointer this service owns: PostgreSQL when DB_URI is set, memory when
# it isn't. Chapter 9's rule is back because no server is storing threads.
#
# Before you run this:
# 1. From 13-Agent-Server/pending-command-demo/:
#    python -m pip install -e ".[backend]"
# 2. Optional, so threads survive a restart (Appendix F sets up the database):
#    export DB_URI="postgresql://<user>@localhost:5432/langgraph_stm?sslmode=disable"
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    uvicorn backend.app:app --port 8000
#
# Expected: the log names the checkpointer, then
#    Uvicorn running on http://127.0.0.1:8000
# The curl commands in README.md call the two endpoints.

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.routes import api_router
from server.graph import builder

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the checkpointer once, compile the graph with it, close it at shutdown."""
    db_uri = os.getenv("DB_URI")
    if db_uri:
        async with AsyncPostgresSaver.from_conn_string(db_uri) as saver:
            await saver.setup()  # creates the checkpoint tables; safe to repeat
            app.state.graph = builder.compile(checkpointer=saver)
            logger.info("Checkpointer: PostgreSQL")
            yield
    else:
        # Fine for a first run. Every thread is gone when the process stops.
        app.state.graph = builder.compile(checkpointer=InMemorySaver())
        logger.warning("Checkpointer: memory. Set DB_URI to keep threads across restarts.")
        yield


app = FastAPI(title="Pending commands", lifespan=lifespan)
app.include_router(api_router)
