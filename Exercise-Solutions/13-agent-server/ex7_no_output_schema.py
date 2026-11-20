# Exercise 16.7 — break it on purpose: build the gateway without output_schema.
#
# The same dispatch node and input schema as server/gateway.py, compiled with
# no output_schema. The caller now gets the whole thread state back: the
# request it sent, the handler's own keys, and ping_count.
#
# Why this matters:
# - output_schema is the gateway's response contract. Without it, every key
#   you add to the state later leaks to every caller.
# - The input side still holds: GatewayRequest decides what a caller may write.
#
# This runs the graph in-process with a memory checkpointer, so no server is
# needed. On LangGraph Server, deleting output_schema=GatewayResponse from
# server/gateway.py and calling GET /status gives the same reply.
#
# Before you run this:
#    python -m pip install -e .
#    from 13-Agent-Server/pending-command-demo/
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    python ../../Exercise-Solutions/13-agent-server/ex7_no_output_schema.py
#
# Expected:
#    POST /ping -> {'method': 'POST', 'path': '/ping', 'body': None, 'status': 204, 'ping_count': 1, 'response': None}
#    GET /status -> {'method': 'GET', 'path': '/status', 'body': {'status': 'ok', 'ping_count': 1}, 'status': 200, 'ping_count': 1, 'response': {'status': 'ok', 'ping_count': 1}}

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from server.gateway import GatewayRequest, GatewayState, dispatch

# Exactly the builder in server/gateway.py, minus output_schema.
builder = StateGraph(GatewayState, input_schema=GatewayRequest)
builder.add_node("dispatch", dispatch)
builder.add_edge(START, "dispatch")
builder.add_edge("dispatch", END)
# In-process, so this file has to store the thread itself (Chapter 9).
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "exercise-16-7"}}
for method, path in [("POST", "/ping"), ("GET", "/status")]:
    reply = graph.invoke({"method": method, "path": path, "body": None}, config)
    print(f"{method} {path} -> {reply}")
