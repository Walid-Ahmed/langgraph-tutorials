# A simple chatbot with both memory scopes:
#
# - MemorySaver remembers messages inside one thread_id.
# - InMemoryStore remembers one user_details entry across thread_ids.
#
# This example intentionally stores one plain text memory. The next example,
# 02_structured_cross_thread_memory.py, adds structured extraction and safer
# merging.
#
# Run from the repository root (requires OPENAI_API_KEY in .env or the shell):
#   python "8-Long-Term-Memory/01_simple_cross_thread_memory.py"

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from util import plot_graph  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "Missing OPENAI_API_KEY.\n"
        f"Add OPENAI_API_KEY=your_key_here to:\n{REPO_ROOT / '.env'}"
    )

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


@dataclass
class Context:
    # @dataclass automatically creates Context.__init__(user_id), so we can
    # construct the object with Context(user_id="user-1") without writing an
    # __init__ method ourselves. It also gives the object a readable __repr__.
    #
    # user_id: str declares the one field that this runtime context carries.
    # The type annotation helps editors and type checkers; Python does not
    # normally enforce it at runtime.
    #
    # user_id identifies a PERSON, not a conversation. The same user_id can
    # read the same Store memory from several different chats. It describes
    # who is running the graph, not which chat is running.
    user_id: str


def chat(state: MessagesState, runtime: Runtime[Context]) -> dict:
    """Read one saved user_details memory, then answer the user."""
    # Store entries are addressed by a namespace (like a folder) and a key
    # (like a filename). This namespace gives each user a separate folder.
    namespace = ("memory", runtime.context.user_id)
    saved_item = runtime.store.get(namespace, "user_details")
    # get() returns an Item. The saved dictionary is available in Item.value.
    saved_details = (
        saved_item.value["memory"]
        if saved_item
        else "No existing user details found."
    )

    # FIRST LLM CALL: answer using long-term Store facts plus the short-term
    # messages MemorySaver loaded for this thread_id.
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant with memory. Personalize your answer "
            "only with facts found in the saved user details.\n\n"
            f"Saved user details:\n{saved_details}"
        )
    )
    response = model.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


def update_memory(state: MessagesState, runtime: Runtime[Context]) -> dict:
    """Ask the LLM to merge new user facts into one plain text memory."""
    # This node runs after chat and reads the same user profile before updating it.
    namespace = ("memory", runtime.context.user_id)
    key = "user_details"
    saved_item = runtime.store.get(namespace, key)
    saved_details = (
        saved_item.value["memory"]
        if saved_item
        else "No existing user details found."
    )

    # Use only the latest human message. This keeps assistant-generated claims
    # out of the extraction prompt.
    latest_user_message = next(
        message
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )
    extraction_prompt = SystemMessage(
        content=(
            "Maintain a concise bulleted user profile.\n\n"
            f"CURRENT PROFILE:\n{saved_details}\n\n"
            "Read the latest user message. Keep existing facts, add or update "
            "only facts explicitly stated by the user, and return the complete "
            "updated profile. If the message contains no personal facts, "
            "return the current profile unchanged."
        )
    )
    # SECOND LLM CALL: this does not answer the user. It only creates the
    # updated profile that will be written to long-term memory.
    updated_profile = model.invoke([extraction_prompt, latest_user_message])

    # put() creates this entry on the first run. Later calls with the same
    # namespace and key replace its value with the newly merged profile.
    runtime.store.put(namespace, key, {"memory": updated_profile.content})
    return {}


def build_graph():
    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node("chat", chat)
    builder.add_node("update_memory", update_memory)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", "update_memory")
    builder.add_edge("update_memory", END)
    return builder


def run_thread(graph, thread_id: str, context: Context, message: str) -> None:
    """Run one thread and print the assistant response."""
    # thread_id identifies a CHAT. MemorySaver keeps each thread history separate.
    config = {"configurable": {"thread_id": thread_id}}
    final_state = None

    for event in graph.stream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        context=context,
        stream_mode="values",
    ):
        final_state = event

    assert final_state is not None
    print(f"\nThread: {thread_id}")
    print(f"User: {message}")
    print(f"Assistant: {final_state['messages'][-1].content}")


def show_saved_memory(store: InMemoryStore, user_id: str) -> None:
    """Print the exact long-term memory stored for this user."""
    item = store.get(("memory", user_id), "user_details")
    print("\nStored long-term memory:")
    print(item.value["memory"] if item else "No memory saved.")


def main() -> None:
    # Checkpointer = short-term messages scoped to thread_id.
    # Store = long-term user facts scoped here to user_id.
    # Both objects below are in-memory only and disappear when this process ends.
    short_term_memory = MemorySaver()
    long_term_memory = InMemoryStore()
    graph = build_graph().compile(
        checkpointer=short_term_memory,
        store=long_term_memory,
    )

    plot_graph(
        graph,
        REPO_ROOT / "8-Long-Term-Memory/diagrams/simple_cross_thread_memory_graph.png",
    )

    # Reusing user_id lets Thread 2 read facts learned in Thread 1, even though
    # the two thread_id values keep their chat histories separate.
    same_user = Context(user_id="user-1")

    # Thread 1 introduces the user and project, then update_memory stores them.
    run_thread(
        graph,
        thread_id="thread-1",
        context=same_user,
        message=(
            "My name is Walid. I am a software engineer working on a LangGraph "
            "tutorial repository."
        ),
    )
    show_saved_memory(long_term_memory, same_user.user_id)

    # Thread 2 has no access to Thread 1's messages, but the same user_id lets
    # chat read the stored profile. The new role is then merged into memory.
    run_thread(
        graph,
        thread_id="thread-2",
        context=same_user,
        message=(
            "This is a new chat. I am now an engineering manager. What do you "
            "remember about me?"
        ),
    )
    show_saved_memory(long_term_memory, same_user.user_id)


if __name__ == "__main__":
    main()
