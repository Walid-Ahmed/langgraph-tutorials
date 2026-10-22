# Submit PING and print only the thread id.
#
# PING has no application reply. The script still drains the stream, so a
# handler error fails the process instead of looking like success.
#
# Before you run this:
# 1. langgraph dev --no-browser
#    from 13-Agent-Server/pending-command-demo/ (see server/graph.py).
#
# Run it with (from 13-Agent-Server/pending-command-demo/):
#    python clients/ping.py
#
# Then run:
#    export THREAD_ID="<the printed id>"
#    python clients/get_status.py
#
# Expected: one line, THREAD_ID=<uuid>, and exit code 0.
# The server log shows PING handled; ping_count=1 on a fresh thread.

from common import get_thread_id, run_command

thread_id = get_thread_id()
print(f"THREAD_ID={thread_id}")
# No reply is printed on purpose. A non-zero exit is how this script reports failure.
run_command(thread_id, "PING")
