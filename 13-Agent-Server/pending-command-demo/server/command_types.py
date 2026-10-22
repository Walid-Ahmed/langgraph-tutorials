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
    type: Literal["PING", "GET_STATUS"]


class StatusResponse(TypedDict):
    status: Literal["ok"]
    ping_count: int


class State(TypedDict, total=False):
    pending_command: PendingCommand | None
    response: StatusResponse | None
    ping_count: int
