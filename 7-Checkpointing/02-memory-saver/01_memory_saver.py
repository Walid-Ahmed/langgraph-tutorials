# Same chatbot as 00_no_memory.py, but compiled WITH a MemorySaver
# checkpointer and a fixed thread_id. Now a second invoke() on the same
# thread remembers the first turn, proving the checkpointer is what
# provides conversational memory.
#
# Put this in the repository-root .env file:
#   OPENAI_API_KEY=your_key_here
#
# Then run:
#   python "7-Checkpointing/02-memory-saver/01_memory_saver.py"

import os
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from util import plot_graph

# Load the same .env file even when the script is launched by an IDE or with
# an absolute path from a different working directory.
load_dotenv(REPO_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "Missing OPENAI_API_KEY.\n"
        f"Add OPENAI_API_KEY=your_key_here to:\n{REPO_ROOT / '.env'}"
    )

client = OpenAI(api_key=api_key)


class State(TypedDict):
    # `add_messages` appends new messages instead of replacing the list.
    # Because this graph also has a checkpointer, the resulting message list
    # is saved after the node runs and can be restored on the next invoke.
    messages: Annotated[list, add_messages]


def chat_node(state: State):
    # On the second invoke, `state["messages"]` contains:
    # 1. the first user message,
    # 2. the first assistant reply, and
    # 3. the new user question.
    messages_for_openai = []

    for message in state["messages"]:
        # `add_messages` converts dictionaries into LangChain objects:
        # - {"role": "user", ...} becomes HumanMessage(type="human")
        # - {"role": "assistant", ...} becomes AIMessage(type="ai")
        # That is why `message.type` and `message.content` are available.

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

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_for_openai,
        max_tokens=256,
    )

    # `add_messages` appends this reply to the restored message history.
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


# Build a one-node graph:
# START -> chat -> END
builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# MemorySaver stores checkpoints in this Python process's memory. It is useful
# for learning, tests, and short-lived programs, but its data disappears when
# this process stops.
checkpointer = MemorySaver()

# Passing the checkpointer here enables persistence between graph.invoke()
# calls. Without this argument, the graph would behave like 00_no_memory.py.
graph = builder.compile(checkpointer=checkpointer)

plot_graph(graph)

# One thread_id identifies one isolated conversation.
# Reusing this ID restores and extends the same conversation. A different ID
# would start a separate conversation with no knowledge of this one.
config = {
    "configurable": {
        "thread_id": "walid-session",
    }
}

# Run 1 — start the thread and introduce yourself. After the chat node runs,
# MemorySaver saves the user message and assistant reply under walid-session.
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Walid"}]}, config)

# Run 2 — MemorySaver finds walid-session, restores its saved messages, and
# merges this new question into that history before running the chat node.
result = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)
print("Bot:", result["messages"][-1].content)
# Expected meaning: "Your name is Walid." Exact wording may vary.

# get_state() returns the latest StateSnapshot saved for this thread.
# `.values` contains the saved state fields; `.next` contains the node(s)
# scheduled to run next. Here `.next` is empty because the graph reached END.
saved_snapshot = graph.get_state(config)
print("Stored values:", saved_snapshot.values)
print("Next nodes:", saved_snapshot.next)
