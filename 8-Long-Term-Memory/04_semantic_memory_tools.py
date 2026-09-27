# Run from the repository root:
#   python "8-Long-Term-Memory/04_semantic_memory_tools.py"
#
# What it does:
#   Gives an agent two memory tools (LangMem's manage_memory and search_memory)
#   over an embedding-indexed InMemoryStore. In one run the user states facts;
#   in a second, separate run (no shared messages) the agent searches memory to
#   personalize its answer.
#
# What it demonstrates (Chapter 11 — semantic memory):
#   - semantic memory = facts about the user, saved and found BY THE AGENT
#   - hot-path updates: the agent saves memory while it answers
#   - a namespace template, {langgraph_user_id}, filled from config per user
#   - an embedding index so search() matches by meaning, not exact words
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env (chat model + embeddings); langmem.
#
# Expected output (wording varies; the tool calls should not):
#   Run 1 -> one or more manage_memory(action="create") calls, then a short reply
#   Run 2 -> a search_memory call, then a practice question in Python aimed at
#            data-engineering interviews
#   Store inspection -> Maya's facts listed; another user's namespace is empty

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# The index makes the Store embed each saved value, so search(query=...)
# returns the memories closest in meaning to the query.
store = InMemoryStore(index={"embed": "openai:text-embedding-3-small", "dims": 1536})

# {langgraph_user_id} is filled from config["configurable"] on every call,
# so one tool object keeps each user's facts in a separate namespace.
FACTS_NAMESPACE = ("assistant", "{langgraph_user_id}", "facts")

SYSTEM_PROMPT = (
    "You are a helpful study assistant. "
    "When the user states a stable fact about themselves (goals, background, "
    "preferences), save it with manage_memory. "
    "Before answering a request that could depend on who the user is, call "
    "search_memory. Never claim to remember something search_memory did not return."
)


def build_agent():
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[
            create_manage_memory_tool(namespace=FACTS_NAMESPACE),
            create_search_memory_tool(namespace=FACTS_NAMESPACE),
        ],
        system_prompt=SYSTEM_PROMPT,
        store=store,  # both runs must share this Store object
    )


def show_run(title: str, messages: list) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    for message in messages:
        message.pretty_print()


def inspect_store(user_id: str) -> None:
    """Look inside the Store directly: what did the agent decide to save?"""
    namespace = ("assistant", user_id, "facts")
    print("\n" + "=" * 70 + "\nSTORE INSPECTION\n" + "=" * 70)
    for item in store.search(namespace):
        print(f"- {item.key}: {item.value}")
    for item in store.search(namespace, query="programming language"):
        print(f"  search 'programming language' -> score={item.score:.2f} {item.value}")
    others = store.search(("assistant", "another-user", "facts"))
    print(f"- another-user has {len(others)} memories (isolated namespace)")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repository-root .env file.")

    agent = build_agent()
    config = {"configurable": {"langgraph_user_id": "maya"}}

    run1 = agent.invoke(
        {"messages": [{"role": "user", "content":
            "I'm preparing for data-engineering interviews, and I prefer examples in Python."}]},
        config=config,
    )
    show_run("RUN 1 — THE USER STATES FACTS", run1["messages"])

    # A brand-new invocation: no messages are shared with run 1.
    run2 = agent.invoke(
        {"messages": [{"role": "user", "content": "Give me one practice question."}]},
        config=config,
    )
    show_run("RUN 2 — A NEW CONVERSATION USES THEM", run2["messages"])

    inspect_store("maya")


if __name__ == "__main__":
    main()
