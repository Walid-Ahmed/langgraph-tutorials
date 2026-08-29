# Baseline "no memory" example: a graph compiled WITHOUT a checkpointer.
# Each graph.invoke() starts from a clean slate, so a second run has no idea
# what was said in the first — demonstrates why checkpointing is needed.
#
# Run from the repository root (requires OPENAI_API_KEY in the environment):
#   python "7-Checkpointing/02-memory-saver/01_no_memory.py"

import sys
from pathlib import Path
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from openai import OpenAI
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from util import plot_graph

load_dotenv(REPO_ROOT / ".env")

client = OpenAI()


class State(TypedDict):
    # `add_messages` is a reducer. During ONE graph invocation, it appends a
    # node's new message to the messages already present in the current state.
    #
    # Important: a reducer is not persistent memory. Without a checkpointer,
    # this accumulated state is discarded when graph.invoke() finishes.
    messages: Annotated[list, add_messages]


def chat_node(state: State):
    # The node receives the current invocation's complete message history.
    # In this example that history contains only the message supplied to the
    # current graph.invoke(), because no earlier state was checkpointed.
    messages_for_openai = []

    for message in state["messages"]:
        # We did not set `message.type` ourselves. The `add_messages` reducer
        # converts message dictionaries into LangChain message objects:
        # - {"role": "user", ...} becomes HumanMessage(type="human")
        # - {"role": "assistant", ...} becomes AIMessage(type="ai")
        # The message class therefore supplies `message.type` automatically.

        # Translate LangChain's role names into the names OpenAI expects.
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

    # Send this invocation's messages to the model.
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_for_openai,
        max_tokens=256,
    )

    # `add_messages` appends this assistant reply to the current state.
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


# Build a one-node graph:
# START -> chat -> END
builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# No checkpointer is passed to compile(). The graph can process messages, but
# it has nowhere to save state between separate invoke() calls.
graph = builder.compile()

plot_graph(graph)

# Run 1 — introduce yourself.
# The returned state contains both this user message and the assistant reply,
# but we deliberately do not save or pass that result into the next call.
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Walid"}]})

# Run 2 — this is a completely fresh input state. Reusing the same `graph`
# Python object does NOT preserve the first run. Persistence requires either:
# 1. a checkpointer plus a thread_id, or
# 2. manually passing the previous message history into this invocation.
result = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]})
print("Bot:", result["messages"][-1].content)

# Expected meaning: the model cannot know the name from Run 1 because it did
# not receive Run 1's messages. Exact wording can vary between model calls.
# Example: "I don't know your name; you haven't told me."
