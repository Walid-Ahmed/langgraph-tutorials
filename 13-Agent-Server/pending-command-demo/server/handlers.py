# Handlers for the two commands in Chapter 16. graph.py calls these functions.
# They are not graph nodes and not separate servers.
#
# Why this matters:
# - PING has no application reply. It still writes response: None so an older
#   GET_STATUS reply is not left behind looking like this run's answer.
# - ping_count is per thread, so a second client can see that the first run
#   really happened. It is not a server-wide health metric.

import logging

from server.command_types import State

logger = logging.getLogger(__name__)


def handle_ping(state: State) -> State:
    """Increment this thread's counter and clear any previous reply."""
    count = state.get("ping_count", 0) + 1
    logger.info("PING handled; ping_count=%s", count)
    return {"ping_count": count, "response": None}


def handle_get_status(state: State) -> State:
    """Report the counter. A missing counter is 0, not an error."""
    return {
        "response": {
            "status": "ok",
            "ping_count": state.get("ping_count", 0),
        }
    }
