# An alternative to checkpointing: no checkpointer at all — instead the
# caller manually carries the growing "messages" list forward between
# invoke() calls (result["messages"] + [new turn]), showing memory can be
# managed by the caller instead of LangGraph's persistence layer.
#
# Put this in the repository-root .env file:
#   OPENAI_API_KEY=your_key_here
#
# Then run:
#   python "7-Checkpointing/02-memory-saver/03_manual_history.py"

import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from util import plot_graph  # noqa: E402

# Load the repository's .env even when an IDE launches this file from a
# different working directory.
load_dotenv(REPO_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "Missing OPENAI_API_KEY.\n"
        f"Add OPENAI_API_KEY=your_key_here to:\n{REPO_ROOT / '.env'}"
    )

client = OpenAI(api_key=api_key)


class State(TypedDict):
    # `add_messages` appends the chat node's reply to the messages supplied
    # for the CURRENT invoke. It does not persist them after invoke returns.
    messages: Annotated[list, add_messages]


def chat_node(state: State):
    messages_for_openai = []

    for message in state["messages"]:
        # `add_messages` converts dictionaries into LangChain message objects.
        # A user dictionary becomes HumanMessage(type="human"), and an
        # assistant dictionary becomes AIMessage(type="ai").
        openai_role = message.type
        if message.type == "human":
            openai_role = "user"
        elif message.type == "ai":
            openai_role = "assistant"

        message_for_openai = {
            "role": openai_role,
            "content": message.content,
        }
        messages_for_openai.append(message_for_openai)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_for_openai,
        max_tokens=256,
    )

    # The reducer appends this reply to the input messages and returns the
    # complete conversation as part of this invocation's result.
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


# Build a one-node graph:
# START -> chat -> END
builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# There is deliberately no checkpointer. The graph will not load or save any
# conversation automatically.
graph = builder.compile()

plot_graph(graph)

# Run 1 — start with one user message.
first_input = {
    "messages": [
        {"role": "user", "content": "Hi, my name is Walid"},
    ]
}
first_result = graph.invoke(first_input)
print("Bot:", first_result["messages"][-1].content)

# The CALLER saves the returned history. LangGraph does not save it.
conversation_history = first_result["messages"]

# Run 2 — manually send the entire old history plus the new question.
second_input = {
    "messages": conversation_history
    + [
        {"role": "user", "content": "What is my name?"},
    ]
}
second_result = graph.invoke(second_input)
print("Bot:", second_result["messages"][-1].content)

# Save the newly returned history again.
conversation_history = second_result["messages"]

# Run 3 — repeat the same pattern: old history + one new user message.
third_input = {
    "messages": conversation_history
    + [
        {"role": "user", "content": "And say goodbye to me by name!"},
    ]
}
third_result = graph.invoke(third_input)
print("Bot:", third_result["messages"][-1].content)

# Mental model:
# - MemorySaver: LangGraph loads history using a thread_id.
# - Manual history: your application stores and resends the entire list.
