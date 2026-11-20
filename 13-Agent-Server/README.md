# 13 — Agent Server

Serving a graph with LangGraph Server, starting with the smallest command mailbox (book Chapter 16).

| Folder | What it shows |
|---|---|
| [`pending-command-demo/`](pending-command-demo/) | One local server, one node, two commands (`PING`, `GET_STATUS`). The same handlers behind an API gateway graph (`POST /ping`, `GET /status`), and behind a FastAPI service. No API key. |
