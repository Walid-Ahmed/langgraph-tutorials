# Exercise solutions for Chapter 16

The server in `13-Agent-Server/pending-command-demo/` has to be running (`langgraph dev --no-browser` from that folder). Use that folder's virtual environment for the client script.

Exercises 16.4–16.6 need no running server. They import the FastAPI app from `backend/` (install it with `python -m pip install -e ".[backend]"`) and run it in-process with FastAPI's `TestClient`, from `13-Agent-Server/pending-command-demo/`.

| Exercise | File | What it shows |
|---|---|---|
| 16.1 | `ex1_reuse_thread.py` | Two PINGs on one thread, then GET_STATUS, prints `ping_count` 2 |
| 16.2 | `ex2_router_forgets_to_clear.py` | The return line with the clear removed, and the error that produces |
| 16.3 | `ex3_unknown_command.py` | A command type the router does not know |
| 16.4 | `ex4_two_users.py` | Two users on the FastAPI service get separate threads (2 and 0) |
| 16.5 | `ex5_no_checkpointer.py` | The FastAPI graph compiled with no checkpointer: the count stays 0 |
| 16.6 | `ex6_restart.py` | A restart loses the thread in memory and keeps it in PostgreSQL |
