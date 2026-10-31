# Pending commands on LangGraph Server

Companion code for Chapter 16 of *Building AI Agents with LangGraph*.

A client writes a small command into a thread's state. One node handles it and clears the slot. `PING` has no application reply. `GET_STATUS` writes a reply the client reads from the last state snapshot. There is no model and no API key.

`pending_command` is ordinary state. It is not `langgraph.types.Command`.

Python 3.12 is the tested interpreter (the project allows 3.11–3.13). From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
langgraph dev --no-browser
```

Leave that process running. In a second terminal, from this same folder, activate `.venv` and run:

```bash
python clients/ping.py
export THREAD_ID="paste-the-printed-id"
python clients/get_status.py
```

After one `PING` on that thread the status client prints `{"status": "ok", "ping_count": 1}`. A second `PING` on the same `THREAD_ID`, then `GET_STATUS` again, prints `ping_count` 2. Without `THREAD_ID`, each client creates its own thread.

`SERVER_URL` defaults to `http://127.0.0.1:2024`. If that port is taken, start with `langgraph dev --no-browser --port 2025` and set `SERVER_URL=http://127.0.0.1:2025` in the client terminal. Stop the server with Ctrl+C.

## Option B: the same graph behind your own FastAPI service

`backend/` runs the graph inside a FastAPI process instead of on LangGraph Server. It reuses the `builder` from `server/graph.py` and compiles it with a checkpointer the service owns. The browser (here, `curl`) sends a demo bearer token, never a thread id. The service picks one thread per user.

From this folder:

```bash
python -m pip install -e ".[backend]"
uvicorn backend.app:app --port 8000
```

In a second terminal:

```bash
curl -i -X POST localhost:8000/api/ping -H "Authorization: Bearer token-amira"
curl localhost:8000/api/status -H "Authorization: Bearer token-amira"
curl localhost:8000/api/status -H "Authorization: Bearer token-omar"
```

The ping answers `HTTP/1.1 204 No Content`. Amira's status is `{"status":"ok","ping_count":1}` and Omar's is `{"status":"ok","ping_count":0}`. A missing token answers 401 `{"detail":"Not authenticated"}`, and an unknown one 401 `{"detail":"Unknown token"}`.

Without `DB_URI`, threads live in memory and a restart sets every count back to 0. To keep them, point `DB_URI` at the database from Appendix F before starting uvicorn:

```bash
export DB_URI="postgresql://<user>@localhost:5432/langgraph_stm?sslmode=disable"
```

The startup log says which checkpointer is in use (`Checkpointer: PostgreSQL` or `Checkpointer: memory`).
