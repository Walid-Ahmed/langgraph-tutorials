# Exercise 16.2 — the router returns the handler update and does not clear the slot.
#
# This file is the broken return. It is not registered with the server.
# To see the failure, replace the return in server/graph.py with broken_return,
# restart or let `langgraph dev` reload, and run clients/ping.py. Put the
# original return back afterwards.
#
# Why this matters:
# - A key the node does not return keeps the value the input wrote.
# - The client treats a leftover command as a failed run.
#
# Before you run the broken graph:
# 1. langgraph dev --no-browser
#    from 13-Agent-Server/pending-command-demo/
#
# Expected when clients/ping.py hits the broken router:
#    RuntimeError: Command was not cleared
# The last values snapshot still holds pending_command {"type": "PING"}.

def broken_return(update: dict) -> dict:
    """What Exercise 16.2 asks you to return instead of the clear."""
    return update
