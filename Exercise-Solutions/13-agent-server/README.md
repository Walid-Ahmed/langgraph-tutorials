# Exercise solutions for Chapter 16

The server in `13-Agent-Server/pending-command-demo/` has to be running (`langgraph dev --no-browser` from that folder). Use that folder's virtual environment for the client script.

| Exercise | File | What it shows |
|---|---|---|
| 16.1 | `ex1_reuse_thread.py` | Two PINGs on one thread, then GET_STATUS, prints `ping_count` 2 |
| 16.2 | `ex2_router_forgets_to_clear.py` | The return line with the clear removed, and the error that produces |
| 16.3 | `ex3_unknown_command.py` | A command type the router does not know |
