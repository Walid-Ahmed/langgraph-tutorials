# A one-node graph that answers HTTP-style requests: Chapter 16's API gateway.
#
# LangGraph Server gives every graph the same run URLs. This graph puts your
# own routes behind them. The client sends a method, a path and a body as the
# run's input. One node, dispatch, looks the pair up in ROUTES, checks the
# body, calls the same handlers the command mailbox uses, and returns a status
# code and a body.
#
# Why this matters:
# - The route table is a dictionary, like HANDLERS in graph.py. The lookup key
#   is (method, path) instead of a command type.
# - 404, 405 and 422 are returned as data. The handler never runs for them.
# - input_schema and output_schema are the contract: a caller can set only
#   method, path and body, and reads back only status and body. ping_count
#   stays inside the thread.
#
# Before you run this:
# 1. From 13-Agent-Server/pending-command-demo/, install the project:
#       python -m pip install -e .
# 2. langgraph.json registers this graph as "gateway", next to "commands".
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    langgraph dev --no-browser
#
# Then run:
#    python clients/gateway_call.py

from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import TypedDict

from server.command_types import StatusResponse
from server.handlers import handle_get_status, handle_ping


class GatewayRequest(TypedDict):
    """What a caller may send. Any other input key is dropped."""

    method: str
    path: str
    body: dict[str, Any] | None


class GatewayResponse(TypedDict):
    """What a caller gets back, shaped like an HTTP response."""

    status: int
    body: Any


class GatewayState(TypedDict, total=False):
    method: str
    path: str
    body: Any  # the request body on the way in, the reply body on the way out
    status: int
    # The handlers from handlers.py read and write these two keys.
    ping_count: int
    response: StatusResponse | None


class NoBody(BaseModel):
    """Both routes take no body. An unknown field is an error, not ignored."""

    model_config = ConfigDict(extra="forbid")


# (method, path) -> (body model, handler). A new endpoint is one more entry.
ROUTES = {
    ("POST", "/ping"): (NoBody, handle_ping),
    ("GET", "/status"): (NoBody, handle_get_status),
}


def dispatch(state: GatewayState) -> GatewayState:
    """Find the route, check the body, call the handler, shape the reply."""
    key = (state["method"].upper(), state["path"])
    if key not in ROUTES:
        allowed = sorted(method for method, path in ROUTES if path == key[1])
        if allowed:
            return {"status": 405, "body": {"detail": "Method not allowed", "allowed": allowed}}
        return {"status": 404, "body": {"detail": "Not found"}}

    body_model, handler = ROUTES[key]
    try:
        body_model.model_validate(state.get("body") or {})
    except ValidationError as err:
        # Same status FastAPI uses for a body that fails its model.
        return {"status": 422, "body": {"detail": err.errors(include_url=False)}}

    update = handler(state)
    reply = update.get("response")
    # PING has no reply, so it answers 204, the way /api/ping does in backend/.
    return {**update, "status": 200 if reply is not None else 204, "body": reply}


builder = StateGraph(
    GatewayState,
    input_schema=GatewayRequest,
    output_schema=GatewayResponse,
)
builder.add_node("dispatch", dispatch)
builder.add_edge(START, "dispatch")
builder.add_edge("dispatch", END)
# As with graph.py, the server stores the thread. No checkpointer here.
graph = builder.compile()
