# Call the API gateway graph with six HTTP-style requests on one thread.
#
# Every request goes to the same server URL, POST /threads/{id}/runs/wait,
# with assistant_id "gateway". The method, path and body travel inside the
# input. The reply is the graph's output: a status code and a body.
#
# Why this matters:
# - runs.wait returns the final output in one response, so there is no
#   first snapshot to skip (compare common.py, which streams).
# - body is sent on every request, even when it's None. A key left out of
#   the input keeps the value the last run on this thread gave it.
#
# Before you run this:
# 1. langgraph dev --no-browser
#    from 13-Agent-Server/pending-command-demo/.
# 2. Optional: THREAD_ID to continue a thread, SERVER_URL for another port.
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    python clients/gateway_call.py
#
# Expected on a fresh thread:
#    POST /ping -> 204 None
#    GET /status -> 200 {'status': 'ok', 'ping_count': 1}
#    GET /ping -> 405 {'detail': 'Method not allowed', 'allowed': ['POST']}
#    GET /pings -> 404 {'detail': 'Not found'}
#    POST /ping {'ping_count': 99} -> 422 {'detail': [...extra_forbidden...]}
#    GET /status -> 200 {'status': 'ok', 'ping_count': 1}

from common import client, get_thread_id


def call(thread_id: str, method: str, path: str, body: dict | None = None) -> dict:
    """Send one request through the gateway and return its status and body."""
    return client.runs.wait(
        thread_id,
        "gateway",  # the second graph name in langgraph.json
        input={"method": method, "path": path, "body": body},
    )


thread_id = get_thread_id()
print(f"THREAD_ID={thread_id}")
requests = [
    ("POST", "/ping", None),
    ("GET", "/status", None),
    ("GET", "/ping", None),
    ("GET", "/pings", None),
    ("POST", "/ping", {"ping_count": 99}),  # trying to set the counter directly
    ("GET", "/status", None),
]
for method, path, body in requests:
    reply = call(thread_id, method, path, body)
    shown = f" {body}" if body else ""
    print(f"{method} {path}{shown} -> {reply['status']} {reply['body']}")
