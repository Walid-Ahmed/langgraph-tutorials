# The two URLs a front-end button calls: POST /api/ping and GET /api/status.
#
# Each endpoint checks the user, picks that user's thread, and runs the same
# graph LangGraph Server ran in the first half of Chapter 16. The browser
# never sends a thread id and never sees the graph state. GET /api/status
# returns only the reply, shaped by a Pydantic model.
#
# Before you run this:
#    python -m pip install -e ".[backend]"
#    from 13-Agent-Server/pending-command-demo/
#
# It is served by backend/app.py:
#    uvicorn backend.app:app --port 8000
#
# Expected (curl commands in README.md): POST /api/ping answers 204 with no
# body. GET /api/status then answers {"status":"ok","ping_count":1} for the
# same user, and each user has a separate count.

from typing import Literal

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from backend.auth import current_user

# Named api_router so it isn't confused with the graph's router node.
api_router = APIRouter(prefix="/api")


class StatusOut(BaseModel):
    """What the browser gets back. Not the graph state, only the reply."""

    status: Literal["ok"]
    ping_count: int


def thread_for(user_id: str) -> str:
    """One thread per user, chosen here, so a browser can't pick someone else's."""
    return f"commands-{user_id}"


async def run_command(request: Request, user_id: str, command_type: str) -> dict:
    """Run one command on the user's thread and return the final state."""
    graph = request.app.state.graph
    # ainvoke returns only the final state, so there is no input snapshot to skip.
    final_state = await graph.ainvoke(
        {"pending_command": {"type": command_type}, "response": None},
        config={"configurable": {"thread_id": thread_for(user_id)}},
    )
    if final_state.get("pending_command") is not None:
        raise RuntimeError("Command was not cleared")
    return final_state


@api_router.post("/ping", status_code=status.HTTP_204_NO_CONTENT)
async def ping(request: Request, user_id: str = Depends(current_user)) -> None:
    await run_command(request, user_id, "PING")


@api_router.get("/status", response_model=StatusOut)
async def get_status(request: Request, user_id: str = Depends(current_user)) -> StatusOut:
    state = await run_command(request, user_id, "GET_STATUS")
    return StatusOut(**state["response"])
