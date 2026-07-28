# Fourth Email Assistant lesson: save and retrieve semantic memory.
#
# The same InMemoryStore is shared by two separate agent invocations:
#   1. "Jim is my friend" -> manage_memory
#   2. "Who is Jim?"      -> search_memory
#
# InMemoryStore shares memories across invocations while this Python process is
# running, but it does not survive a process restart.
#
# Run from the repository root:
#   python "9-Email-Assistant/04_semantic_memory.py"

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from email_assistant.prompts import agent_system_prompt_memory

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": (
        "Senior software engineer leading a team of 5 developers"
    ),
}

prompt_instructions = {
    "agent_instructions": (
        "Use the available tools when appropriate to manage John's tasks "
        "efficiently. Proactively use manage_memory when the user explicitly "
        "provides a stable fact that will be useful in future conversations. "
        "Use search_memory when a request may depend on a previously stored "
        "fact. Never claim to remember something that the memory search did "
        "not return."
    )
}


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Simulate writing and sending an email."""
    return (
        f"SIMULATION ONLY: no email was sent. Drafted email to {to} with "
        f"subject {subject!r}. Content: {content}"
    )


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: str,
) -> str:
    """Simulate scheduling a calendar meeting."""
    return (
        "SIMULATION ONLY: no event was created. "
        f"Proposed {duration_minutes}-minute meeting {subject!r} on "
        f"{preferred_day} with {len(attendees)} attendees."
    )


@tool
def check_calendar_availability(day: str) -> str:
    """Return simulated calendar availability for a given day."""
    return f"SIMULATION ONLY: available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


# The embedding index enables similarity search over saved memory content.
# GPT-4o-mini decides what to remember and when to search; the embedding model
# only converts stored text and queries into vectors for similarity matching.
store = InMemoryStore(
    index={
        "embed": "openai:text-embedding-3-small",
        "dims": 1536,
    }
)

# {langgraph_user_id} is filled from config at invocation time. This keeps each
# user's memory in a separate namespace.
memory_namespace = (
    "email_assistant",
    "{langgraph_user_id}",
    "collection",
)
manage_memory_tool = create_manage_memory_tool(namespace=memory_namespace)
search_memory_tool = create_search_memory_tool(namespace=memory_namespace)


def build_agent():
    """Create the response agent with communication and memory tools."""
    # LangMem turns Store operations into normal agent tools. The agent never
    # accesses the Store directly; it requests manage/search tool calls.
    tools = [
        write_email,
        schedule_meeting,
        check_calendar_availability,
        manage_memory_tool,
        search_memory_tool,
    ]
    system_prompt = agent_system_prompt_memory.format(
        instructions=prompt_instructions["agent_instructions"],
        profile=profile,
        **profile,
    )
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt=system_prompt,
        # Both invocations must receive this exact Store object or the second
        # invocation would have nothing to recall.
        store=store,
    )


def save_trace(runs: list[tuple[str, list]]) -> Path:
    """Save both invocation traces to one local log file."""
    log_path = REPO_ROOT / "9-Email-Assistant/logs/04_semantic_memory_messages.log"
    log_path.parent.mkdir(exist_ok=True)

    sections = []
    for title, messages in runs:
        trace = "\n\n".join(message.pretty_repr() for message in messages)
        sections.append(f"=== {title} ===\n\n{trace}")

    log_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return log_path


def print_trace(title: str, messages: list) -> None:
    """Print one complete agent interaction."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for message in messages:
        message.pretty_print()


def inspect_store(user_id: str, query: str) -> None:
    """Demonstrate direct Store inspection after the agent uses memory tools."""
    namespace = ("email_assistant", user_id, "collection")

    print("\n" + "=" * 70)
    print("DIRECT INMEMORYSTORE INSPECTION")
    print("=" * 70)

    # 1. Discover which memory namespaces currently exist.
    print("\n1. Existing namespaces")
    for existing_namespace in store.list_namespaces():
        print(f"- {tuple(existing_namespace)}")

    # 2. List every memory in this user's namespace without similarity ranking.
    all_memories = store.search(namespace)
    print(f"\n2. All memories for user_id={user_id!r}")
    for item in all_memories:
        print(f"- key={item.key}")
        print(f"  value={item.value}")

    # 3. Run the same kind of semantic search used by search_memory.
    matches = store.search(namespace, query=query)
    print(f"\n3. Semantic search for {query!r}")
    for item in matches:
        print(f"- score={item.score:.3f} key={item.key}")
        print(f"  value={item.value}")

    # 4. Retrieve one exact item when its key is already known.
    if matches:
        selected_key = matches[0].key
        exact_item = store.get(namespace, selected_key)
        print(f"\n4. Exact lookup for key={selected_key!r}")
        print(f"- {exact_item.value if exact_item else 'not found'}")

    # 5. Prove that a different user_id has a separate namespace.
    other_namespace = ("email_assistant", "another-user", "collection")
    other_user_memories = store.search(other_namespace)
    print("\n5. Different-user isolation")
    print(f"- namespace: {other_namespace}")
    print(f"- memories found: {len(other_user_memories)}")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    response_agent = build_agent()
    config = {"configurable": {"langgraph_user_id": "john"}}

    remember_response = response_agent.invoke(
        {"messages": [{"role": "user", "content": "Jim is my friend."}]},
        config=config,
    )
    print_trace("INVOCATION 1 — SAVE A FACT", remember_response["messages"])

    recall_response = response_agent.invoke(
        {"messages": [{"role": "user", "content": "Who is Jim?"}]},
        config=config,
    )
    print_trace("INVOCATION 2 — RECALL THE FACT", recall_response["messages"])

    inspect_store(user_id="john", query="Jim")

    trace_path = save_trace(
        [
            ("INVOCATION 1 — SAVE A FACT", remember_response["messages"]),
            ("INVOCATION 2 — RECALL THE FACT", recall_response["messages"]),
        ]
    )
    print(f"\nInteraction log saved to: {trace_path}")


if __name__ == "__main__":
    main()
