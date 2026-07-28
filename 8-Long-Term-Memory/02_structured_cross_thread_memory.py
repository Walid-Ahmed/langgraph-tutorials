# Demonstrates the two memory scopes working together:
#
# - MemorySaver + thread_id keeps the messages for one conversation.
# - InMemoryStore + user_id shares selected facts across conversations.
#
# The script starts two different threads for the same user. Thread 2 does not
# receive Thread 1's messages, but it can still read the profile saved in the
# shared Store.
#
# Run from the repository root (requires OPENAI_API_KEY in .env or the shell):
#   python "8-Long-Term-Memory/02_structured_cross_thread_memory.py"

import json
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
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from util import plot_graph  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


@dataclass
class Context:
    # user_id selects long-term memory shared across this user's threads.
    user_id: str


class MemoryUpdate(BaseModel):
    """New facts explicitly stated by the user in the latest message."""

    name: str | None = None
    role: str | None = None
    preferences: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)


def build_graph():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    memory_extractor = model.with_structured_output(MemoryUpdate)

    def chat(state: MessagesState, runtime: Runtime[Context]) -> dict:
        """Read this user's long-term profile before producing a response."""
        namespace = (runtime.context.user_id, "memories")
        profile_item = runtime.store.get(namespace, "profile")
        profile = profile_item.value if profile_item else {}

        profile_text = (
            json.dumps(profile, indent=2)
            if profile
            else "No long-term profile has been saved for this user."
        )
        system_message = SystemMessage(
            content=(
                "You are a helpful assistant. Use the saved user profile when "
                "it is relevant, but do not invent missing facts.\n\n"
                f"Saved user profile:\n{profile_text}"
            )
        )
        response = model.invoke([system_message, *state["messages"]])
        return {"messages": [response]}

    def update_memory(state: MessagesState, runtime: Runtime[Context]) -> dict:
        """Extract new user facts and merge them into long-term memory."""
        namespace = (runtime.context.user_id, "memories")
        existing_item = runtime.store.get(namespace, "profile")
        existing = dict(existing_item.value) if existing_item else {}

        latest_user_message = next(
            message
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        update = memory_extractor.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract only personal facts the user explicitly states "
                        "in the message. Do not infer facts from questions or "
                        "from the assistant response. Return empty fields when "
                        "there is nothing new."
                    )
                ),
                latest_user_message,
            ]
        )

        merged = dict(existing)
        if update.name:
            merged["name"] = update.name
        if update.role:
            merged["role"] = update.role

        for field_name in ("preferences", "goals"):
            old_values = list(merged.get(field_name, []))
            new_values = getattr(update, field_name)
            combined = list(dict.fromkeys([*old_values, *new_values]))
            if combined:
                merged[field_name] = combined

        if merged != existing:
            # Reusing this namespace + key updates the user's profile.
            runtime.store.put(namespace, "profile", merged)
            print(f"Saved long-term profile: {merged}")
        else:
            print("No new long-term memory to save.")

        # The Store was updated directly, so this node has no graph-state
        # update to return.
        return {}

    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node("chat", chat)
    builder.add_node("update_memory", update_memory)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", "update_memory")
    builder.add_edge("update_memory", END)

    return builder


def run_thread(graph, thread_id: str, context: Context, message: str) -> None:
    """Stream one conversation thread and print its final response."""
    config = {"configurable": {"thread_id": thread_id}}

    # stream_mode="values" yields the complete state after each graph step.
    # Keeping the last event gives us the final state after update_memory.
    final_state = None
    for event in graph.stream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        context=context,
        stream_mode="values",
    ):
        final_state = event

    assert final_state is not None
    assistant_content = final_state["messages"][-1].content
    print(f"\nThread {thread_id}")
    print(f"User: {message}")
    print(f"Assistant: {assistant_content}")


def show_profile(store: InMemoryStore, user_id: str) -> None:
    """Inspect the exact Store item saved for one user."""
    item = store.get((user_id, "memories"), "profile")
    print(f"\nStored profile for user_id={user_id!r}:")
    if item:
        print(f"- namespace:  {item.namespace}")
        print(f"- key:        {item.key}")
        print(f"- value:      {item.value}")
        print(f"- created_at: {item.created_at}")
        print(f"- updated_at: {item.updated_at}")
    else:
        print("- no profile saved")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY.\n"
            f"Add OPENAI_API_KEY=your_key_here to:\n{REPO_ROOT / '.env'}"
        )

    # Both objects are supplied at compile time, but they have different jobs.
    checkpointer = MemorySaver()
    store = InMemoryStore()
    graph = build_graph().compile(checkpointer=checkpointer, store=store)

    plot_graph(
        graph,
        REPO_ROOT / "8-Long-Term-Memory/diagrams/structured_cross_thread_memory_graph.png",
    )

    walid = Context(user_id="walid")

    # Thread 1 creates short-term message history and saves selected profile
    # facts into long-term memory under user_id="walid".
    run_thread(
        graph,
        thread_id="walid-chat-1",
        context=walid,
        message=(
            "Hi, my name is Walid. I am a software engineer learning LangGraph, "
            "and I prefer concise, step-by-step explanations."
        ),
    )
    show_profile(store, walid.user_id)

    # Thread 2 is a fresh conversation: it has no access to Thread 1's message
    # history. It does share the same Store namespace because user_id is still
    # "walid", so the chat node can recall Walid's profile. It also supplies a
    # newer role, which update_memory merges into that same profile key.
    run_thread(
        graph,
        thread_id="walid-chat-2",
        context=walid,
        message=(
            "This is a new chat. I am now an engineering manager. What do you "
            "remember about me, and how should I approach my new role?"
        ),
    )
    show_profile(store, walid.user_id)

    # A different user_id gets an isolated namespace even when using the same
    # Store instance.
    run_thread(
        graph,
        thread_id="guest-chat-1",
        context=Context(user_id="guest"),
        message="What do you remember about me?",
    )
    show_profile(store, "guest")


if __name__ == "__main__":
    main()
