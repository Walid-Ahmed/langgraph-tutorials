# Exercise 16.4 — two users, two threads, one FastAPI service.
#
# Amira pings twice. Then both users ask for their status. The service picks
# each user's thread from the token, so the counts never mix.
#
# Why this matters:
# - The browser never names a thread, so it can't read or bump someone else's.
# - The graph is unchanged. Only the thread_id the backend passes differs.
#
# Before you run this:
#    python -m pip install -e ".[backend]"
#    from 13-Agent-Server/pending-command-demo/
#
# Run it with (from 13-Agent-Server/pending-command-demo/, DB_URI unset):
#    python ../../Exercise-Solutions/13-agent-server/ex4_two_users.py
#
# Expected:
#    amira {'status': 'ok', 'ping_count': 2}
#    omar {'status': 'ok', 'ping_count': 0}

from fastapi.testclient import TestClient

from backend.app import app

AMIRA = {"Authorization": "Bearer token-amira"}
OMAR = {"Authorization": "Bearer token-omar"}

# TestClient runs the lifespan, so the graph is compiled exactly as under uvicorn.
with TestClient(app) as client:
    client.post("/api/ping", headers=AMIRA)
    client.post("/api/ping", headers=AMIRA)
    print("amira", client.get("/api/status", headers=AMIRA).json())
    print("omar", client.get("/api/status", headers=OMAR).json())
