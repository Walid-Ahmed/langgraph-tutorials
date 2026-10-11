# Step 4 in 2-Reducer: what add_messages does besides appending.
#
# This script calls the add_messages reducer directly, with no graph and no
# model, on a two-message history. It shows the three things the reducer can
# do with an incoming message: append it, replace an existing message that
# has the same id, or remove a message.
#
# Why this matters:
# - 03_messages_reducer.py showed add_messages appending. operator.add can do
#   that too. Replace-by-id and remove are what operator.add cannot do, and
#   they are how later chapters edit or trim a conversation in place.
# - A reducer is an ordinary function (current, new) -> combined, so you can
#   call it yourself to see exactly what it will do inside a graph.
# - It does NOT run a graph or call a model, so no API key is needed.
#
# Before you run this:
# 1. pip install -r requirements.txt   (langgraph and langchain-core; no API key)
#
# Run it from the repository root with:
#    python "2-Reducer/04_messages_update_by_id.py"

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages

history = [
    HumanMessage("My name is Alex.", id="1"),
    AIMessage("Hello Alex.", id="2"),
]


def show(label: str, messages: list) -> None:
    print(f"{label:<9}", [(m.id, m.content) for m in messages])


# A new id: the message is appended.
show("append:", add_messages(history, [HumanMessage("What is RAG?", id="3")]))

# An id that already exists: the old message is replaced, not duplicated.
show("replace:", add_messages(history, [AIMessage("Hello, Alex! How can I help?", id="2")]))

# RemoveMessage with an existing id: that message is deleted.
show("remove:", add_messages(history, [RemoveMessage(id="1")]))
