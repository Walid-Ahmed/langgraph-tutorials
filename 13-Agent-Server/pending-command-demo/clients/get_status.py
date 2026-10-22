# Submit GET_STATUS and print the response object from the final snapshot.
#
# Before you run this:
# 1. langgraph dev --no-browser
#    from 13-Agent-Server/pending-command-demo/.
# 2. To read a thread that already ran PING:
#       export THREAD_ID="<id printed by clients/ping.py>"
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    python clients/get_status.py
#
# Expected after one PING on that thread:
#    {"status": "ok", "ping_count": 1}
# On a brand-new thread the count is 0.

import json

from common import get_thread_id, run_command

thread_id = get_thread_id()
print(f"THREAD_ID={thread_id}")
state = run_command(thread_id, "GET_STATUS")
response = state.get("response")
if response is None:
    raise RuntimeError("No status response received")
print(json.dumps(response, indent=2))
