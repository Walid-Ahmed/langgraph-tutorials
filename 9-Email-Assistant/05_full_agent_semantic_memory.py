# Fifth Email Assistant lesson: combine triage, response tools, and semantic
# memory in one complete graph.
#
# The first email creates a memory about Alice's API-documentation question.
# A separate follow-up invocation searches that memory before drafting a reply.
# All email and calendar tools remain simulations.
#
# Run from the repository root:
#   python "9-Email-Assistant/05_full_agent_semantic_memory.py"

import os
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from email_assistant.prompts import (
    agent_system_prompt_memory,
    triage_system_prompt,
    triage_user_prompt,
)

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
    "triage_rules": {
        "ignore": (
            "Marketing newsletters, spam emails, and mass company announcements"
        ),
        "notify": (
            "Team member out sick, build system notifications, and project "
            "status updates"
        ),
        "respond": (
            "Direct questions from team members, meeting requests, and "
            "critical bug reports"
        ),
    },
    "agent_instructions": (
        "The email and calendar tools in this tutorial are simulations. Never "
        "claim that a real email was sent or a real event was created. Before "
        "answering a follow-up, use search_memory to find relevant earlier "
        "context. Treat questions as questions, not facts: asking whether "
        "something was intentional does not establish that it was or was not "
        "intentional. Never invent a decision, investigation, delegation, "
        "action, or status update that is not present in the current email or "
        "retrieved memory. In this tutorial no internal project status is "
        "available, so acknowledge unresolved questions and say that no "
        "verified update is available. After handling an email, use "
        "manage_memory to store or update the sender, topic, request details, "
        "and only actions actually represented by tool results."
    ),
}


class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


class Router(BaseModel):
    """Analyze an unread email and route it according to its content."""

    reasoning: str = Field(
        description="A concise explanation for the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description=(
            "'ignore' for irrelevant email, 'notify' for important information "
            "that needs no reply, or 'respond' for email that needs a reply."
        )
    )


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Simulate drafting an email without sending it."""
    return (
        "SIMULATION ONLY: no email was sent. "
        f"Drafted email to {to} with subject {subject!r}.\n\n{content}"
    )


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: str,
) -> str:
    """Simulate scheduling a meeting without creating an event."""
    return (
        "SIMULATION ONLY: no event was created. "
        f"Proposed {duration_minutes}-minute meeting {subject!r} on "
        f"{preferred_day} with {len(attendees)} attendees."
    )


@tool
def check_calendar_availability(day: str) -> str:
    """Return simulated calendar availability for a given day."""
    return f"SIMULATION ONLY: available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


store = InMemoryStore(
    index={
        "embed": "openai:text-embedding-3-small",
        "dims": 1536,
    }
)
memory_namespace = (
    "email_assistant",
    "{langgraph_user_id}",
    "collection",
)
manage_memory_tool = create_manage_memory_tool(namespace=memory_namespace)
search_memory_tool = create_search_memory_tool(namespace=memory_namespace)

llm = init_chat_model("openai:gpt-4o-mini")
llm_router = llm.with_structured_output(Router)


def build_response_agent():
    """Create the nested tool-calling agent with semantic-memory access."""
    system_prompt = agent_system_prompt_memory.format(
        instructions=prompt_instructions["agent_instructions"],
        profile=profile,
        **profile,
    )
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[
            write_email,
            schedule_meeting,
            check_calendar_availability,
            manage_memory_tool,
            search_memory_tool,
        ],
        system_prompt=system_prompt,
        store=store,
    )


def triage_router(
    state: State,
) -> Command[Literal["response_agent", "__end__"]]:
    """Classify the email, update messages, and route with Command."""
    email_input = state["email_input"]
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples="No episodic examples are used in this lesson.",
    )
    user_prompt = triage_user_prompt.format(
        author=email_input["author"],
        to=email_input["to"],
        subject=email_input["subject"],
        email_thread=email_input["email_thread"],
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    if result.classification == "respond":
        print("📧 Classification: RESPOND — this email requires a response")
        return Command(
            goto="response_agent",
            update={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Handle this incoming email. Search memory first "
                            "when it appears to be a follow-up. Draft a response "
                            "with the simulated write_email tool, then store or "
                            "update the useful factual context.\n\n"
                            "GROUNDING RULES FOR THIS EXAMPLE:\n"
                            "- The email and retrieved memory are the only facts.\n"
                            "- A sender's question is not evidence of its answer.\n"
                            "- No investigation, documentation decision, or "
                            "project progress has occurred.\n"
                            "- Say that no verified answer or update is available.\n"
                            "- Store the request and simulated draft, not an "
                            "invented answer or status.\n\n"
                            f"{email_input}"
                        ),
                    }
                ]
            },
        )

    if result.classification == "ignore":
        print("🚫 Classification: IGNORE — this email can be safely ignored")
        return Command(goto=END)

    if result.classification == "notify":
        print("🔔 Classification: NOTIFY — this email contains useful information")
        return Command(goto=END)

    raise ValueError(f"Invalid classification: {result.classification}")


def build_email_agent():
    """Compile with thread checkpoints and the nested agent's shared Store."""
    builder = StateGraph(State)
    builder.add_node("triage_router", triage_router)
    builder.add_node("response_agent", build_response_agent())
    builder.add_edge(START, "triage_router")
    return builder.compile(checkpointer=MemorySaver(), store=store)


def render_graph(email_agent) -> None:
    """Print and save the complete graph, including the nested agent."""
    graph_view = email_agent.get_graph(xray=True)
    print("\nMERMAID GRAPH")
    print(graph_view.draw_mermaid())

    output_path = (
        REPO_ROOT
        / "9-Email-Assistant/diagrams/05_full_agent_semantic_memory.png"
    )
    try:
        output_path.write_bytes(graph_view.draw_mermaid_png())
        print(f"\nGraph saved to: {output_path}")
    except Exception as exc:
        print(f"\nPNG rendering skipped: {exc}")


def run_email(
    email_agent,
    email_input: dict,
    config: dict,
    title: str,
) -> list:
    """Run one email and print its complete response-agent trace."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    response = email_agent.invoke(
        {"email_input": email_input, "messages": []},
        config=config,
    )
    messages = response.get("messages", [])
    for message in messages:
        message.pretty_print()
    return messages


def save_trace(runs: list[tuple[str, list]]) -> Path:
    """Save both email-processing traces to one ignored log file."""
    log_path = (
        REPO_ROOT / "9-Email-Assistant/logs/05_full_agent_semantic_memory.log"
    )
    log_path.parent.mkdir(exist_ok=True)
    sections = []
    for title, messages in runs:
        trace = "\n\n".join(message.pretty_repr() for message in messages)
        sections.append(f"=== {title} ===\n\n{trace}")
    log_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return log_path


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    email_agent = build_email_agent()
    render_graph(email_agent)

    # The emails deliberately use different thread_id values, so the follow-up
    # cannot rely on short-term checkpoint history. Both configs use the same
    # langgraph_user_id, allowing semantic facts to cross the thread boundary
    # through InMemoryStore.
    first_config = {
        "configurable": {
            "thread_id": "alice-api-question",
            "langgraph_user_id": "john",
        }
    }
    follow_up_config = {
        "configurable": {
            "thread_id": "alice-api-follow-up",
            "langgraph_user_id": "john",
        }
    }

    first_email = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": (
            "Hi John,\n\nI noticed that /auth/refresh and /auth/validate seem "
            "to be missing from the authentication-service specs. Was that "
            "intentional, or should we update the docs?\n\nThanks!\nAlice"
        ),
    }
    first_messages = run_email(
        email_agent,
        first_email,
        first_config,
        "EMAIL 1 — INITIAL API-DOCUMENTATION QUESTION",
    )

    follow_up_email = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Follow up",
        "email_thread": "Hi John,\n\nAny update on my previous ask?\n\nAlice",
    }
    follow_up_messages = run_email(
        email_agent,
        follow_up_email,
        follow_up_config,
        "EMAIL 2 — FOLLOW-UP REQUIRING MEMORY",
    )

    namespace = ("email_assistant", "john", "collection")
    print("\nSEMANTIC MEMORY AFTER BOTH EMAILS")
    for item in store.search(namespace, query="Alice API documentation"):
        print(f"- score={item.score:.3f} key={item.key}")
        print(f"  {item.value}")

    log_path = save_trace(
        [
            ("EMAIL 1 — INITIAL QUESTION", first_messages),
            ("EMAIL 2 — FOLLOW-UP", follow_up_messages),
        ]
    )
    print(f"\nInteraction log saved to: {log_path}")


if __name__ == "__main__":
    main()
