# Run from the repository root:
#   python "8-Long-Term-Memory/04_semantic_memory_tools.py"
#
# What it does:
#   A study assistant with semantic memory over an embedding-indexed InMemoryStore.
#   In one run the user states facts; in a second, separate run (no shared
#   messages) the assistant uses those facts to personalize its answer.
#
# What it demonstrates (Chapter 11 — semantic memory):
#   - semantic memory = facts about the user
#   - WRITING on the hot path by the agent: LangMem's manage_memory tool, because
#     deciding what is worth remembering is a judgment call the model is good at
#   - READING in code, not by tool: a @dynamic_prompt middleware searches the
#     Store before every model call and puts the facts in the system prompt, so
#     the model cannot "forget to look". (An earlier version gave the model a
#     search_memory tool; in a real run it simply didn't call it in run 2.)
#   - a namespace template, {langgraph_user_id}, filled from config per user
#   - an embedding index so search() matches by meaning, not exact words
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env (chat model + embeddings); langmem.
#
# Expected output (wording varies; the structure should not):
#   Run 1 -> "[memory] recalled 0 fact(s)", one or more manage_memory(action="create")
#            calls, then a short reply
#   Run 2 -> "[memory] recalled 1+ fact(s)", NO tool call, and a practice question
#            in Python aimed at data-engineering interviews
#   Store inspection -> Maya's facts listed; another user's namespace is empty

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langgraph.config import get_config
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool

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
    "Use the facts under 'What you know about this user' to personalize every answer, "
    "and never claim to remember anything that is not listed there."
)


def facts_namespace(user_id: str) -> tuple[str, str, str]:
    return ("assistant", user_id, "facts")


@dynamic_prompt
def recall_facts(request: ModelRequest) -> str:
    """Runs before every model call: read memory in code, so it is never skipped."""
    # Same trusted source the manage_memory tool uses to fill {langgraph_user_id}.
    user_id = get_config()["configurable"]["langgraph_user_id"]
    # Search with what the user just asked; on a tool lap the last message is a
    # ToolMessage, so look back for the most recent human turn.
    question = next((m.content for m in reversed(request.messages) if m.type == "human"), "")
    found = store.search(facts_namespace(user_id), query=question, limit=5)
    print(f"  [memory] recalled {len(found)} fact(s) for {user_id}")
    facts = "\n".join(f"- {item.value['content']}" for item in found) or "- (nothing saved yet)"
    return f"{SYSTEM_PROMPT}\n\nWhat you know about this user:\n{facts}"


def build_agent():
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[create_manage_memory_tool(namespace=FACTS_NAMESPACE)],  # writing stays a tool
        middleware=[recall_facts],                                       # reading is code
        store=store,  # both runs must share this Store object
    )


def show_run(title: str, messages: list) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    for message in messages:
        message.pretty_print()


def inspect_store(user_id: str) -> None:
    """Look inside the Store directly: what did the agent decide to save?"""
    print("\n" + "=" * 70 + "\nSTORE INSPECTION\n" + "=" * 70)
    for item in store.search(facts_namespace(user_id)):
        print(f"- {item.key}: {item.value}")
    for item in store.search(facts_namespace(user_id), query="programming language"):
        print(f"  search 'programming language' -> score={item.score:.2f} {item.value}")
    others = store.search(facts_namespace("another-user"))
    print(f"- another-user has {len(others)} memories (isolated namespace)")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repository-root .env file.")

    agent = build_agent()
    config = {"configurable": {"langgraph_user_id": "maya"}}

    print("RUN 1 — THE USER STATES FACTS")
    run1 = agent.invoke(
        {"messages": [{"role": "user", "content":
            "I'm preparing for data-engineering interviews, and I prefer examples in Python."}]},
        config=config,
    )
    show_run("RUN 1 — MESSAGES", run1["messages"])

    # A brand-new invocation: no messages are shared with run 1.
    print("\nRUN 2 — A NEW CONVERSATION USES THEM")
    run2 = agent.invoke(
        {"messages": [{"role": "user", "content": "Give me one practice question."}]},
        config=config,
    )
    show_run("RUN 2 — MESSAGES", run2["messages"])

    inspect_store("maya")


if __name__ == "__main__":
    main()
