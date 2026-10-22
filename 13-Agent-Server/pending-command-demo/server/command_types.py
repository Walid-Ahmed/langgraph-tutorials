# State for the Chapter 16 command mailbox. Imported by handlers.py and graph.py.
#
# pending_command is one slot of application data. It is not
# langgraph.types.Command, and it is not the server API's top-level command
# field (that one resumes an interrupted run).
#
# total=False means a key may be absent. It does not fill in defaults and it
# does not check incoming JSON at runtime.

from typing import Literal
from typing_extensions import TypedDict


class PendingCommand(TypedDict):
    # The only key the router reads. Extra JSON keys are not part of the contract.
    type: Literal["PING", "GET_STATUS"]


class StatusResponse(TypedDict):
    status: Literal["ok"]
    # Taken from this thread's counter. The handler reports 0 when the key is absent.
    ping_count: int


class State(TypedDict, total=False):
    # One slot. None means empty. Not a queue, and not langgraph.types.Command.
    pending_command: PendingCommand | None
    # Reply for the client. PING writes None so an older status is not this run's answer.
    response: StatusResponse | None
    # Per thread, so a later GET_STATUS can show that an earlier PING really ran.
    ping_count: int
