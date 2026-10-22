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
