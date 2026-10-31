# Exercise 16.6 — restart the service and see which checkpointer kept the thread.
#
# The script starts the app, pings once, stops it, starts it again and asks
# for the status. Run it once without DB_URI and once with it.
#
# Why this matters:
# - InMemorySaver lives in the process. A restart or a second uvicorn worker
#   starts from nothing.
# - PostgresSaver keeps the thread in the database, so the next process,
#   or another worker, picks it up.
#
# Before you run this:
#    python -m pip install -e ".[backend]"
#    from 13-Agent-Server/pending-command-demo/
#    For the second run, the database from Appendix F.
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    python ../../Exercise-Solutions/13-agent-server/ex6_restart.py
#    DB_URI="postgresql://<user>@localhost:5432/langgraph_stm?sslmode=disable" \
#        python ../../Exercise-Solutions/13-agent-server/ex6_restart.py
#
# Expected without DB_URI:
#    before restart {'status': 'ok', 'ping_count': 1}
#    after restart {'status': 'ok', 'ping_count': 0}
# Expected with DB_URI, on a database with no earlier commands-amira thread:
#    before restart {'status': 'ok', 'ping_count': 1}
#    after restart {'status': 'ok', 'ping_count': 1}

from fastapi.testclient import TestClient

from backend.app import app

AMIRA = {"Authorization": "Bearer token-amira"}

# Each with-block is one process lifetime: lifespan opens, then closes, the checkpointer.
with TestClient(app) as client:
    client.post("/api/ping", headers=AMIRA)
    print("before restart", client.get("/api/status", headers=AMIRA).json())

with TestClient(app) as client:
    print("after restart", client.get("/api/status", headers=AMIRA).json())
