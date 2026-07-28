# Eighth Email Assistant lesson: run all memory types in one graph.
#
# One shared InMemoryStore holds three per-user namespaces:
# - collection: semantic facts managed by agent tools
# - examples: episodic triage corrections retrieved with embeddings
# - procedures: exact-key instructions updated from user feedback
#
# MemorySaver separately checkpoints graph state under thread_id.
#
# Run from the repository root:
#   python "9-Email-Assistant/08_integrated_memory_agent.py"

import os
import uuid
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langmem import (
    create_manage_memory_tool,
    create_multi_prompt_optimizer,
    create_search_memory_tool,
)
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

DEFAULT_PROCEDURES = {
    "triage_ignore": "Marketing newsletters, spam, and unsolicited sales messages",
    "triage_notify": (
        "Team member absences, build notifications, and project status updates"
    ),
    "triage_respond": (
        "Direct questions, meeting requests, and critical bug reports"
    ),
    "agent_instructions": (
        "Before answering a follow-up, search semantic memory. After handling "
        "an email, save stable sender, topic, request, and simulated-action facts. "
        "Never invent status or claim that simulated tools made real changes."
    ),
}

PROMPT_SPECS = {
    "agent-instructions": (
        "agent_instructions",
        "Update when feedback changes email writing or calendar-tool behavior.",
    ),
    "triage-ignore": (
        "triage_ignore",
        "Update when feedback changes which emails should be ignored.",
    ),
    "triage-notify": (
        "triage_notify",
        "Update when feedback changes which emails should notify the user.",
    ),
    "triage-respond": (
        "triage_respond",
        "Update when feedback changes which emails require a response.",
    ),
}


class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


class Router(BaseModel):
    reasoning: str = Field(description="A concise reason for the classification.")
    classification: Literal["ignore", "respond", "notify"]


# Semantic and episodic namespaces use this embedding index. Procedural entries
# live in the same Store but are read by exact key, so they do not need search.
store = InMemoryStore(
    index={
        "embed": "openai:text-embedding-3-small",
        "dims": 1536,
    }
)

semantic_namespace = (
    "email_assistant",
    "{langgraph_user_id}",
    "collection",
)
manage_memory_tool = create_manage_memory_tool(namespace=semantic_namespace)
search_memory_tool = create_search_memory_tool(namespace=semantic_namespace)

router_model = init_chat_model("openai:gpt-4o-mini").with_structured_output(Router)
procedure_optimizer = create_multi_prompt_optimizer(
    "openai:gpt-4o-mini",
    kind="prompt_memory",
)


def episodic_namespace(user_id: str) -> tuple[str, str, str]:
    return ("email_assistant", user_id, "examples")


def procedural_namespace(user_id: str) -> tuple[str, str, str]:
    return ("email_assistant", user_id, "procedures")


def initialize_procedures(user_id: str) -> None:
    namespace = procedural_namespace(user_id)
    for key, prompt in DEFAULT_PROCEDURES.items():
        if store.get(namespace, key) is None:
            store.put(
                namespace,
                key,
                {"prompt": prompt, "source": "default"},
                index=False,
            )


def read_procedure(user_id: str, key: str) -> str:
    item = store.get(procedural_namespace(user_id), key)
    if item is None:
        raise KeyError(f"Missing procedure {key!r} for {user_id!r}")
    return item.value["prompt"]


def save_episodic_correction(
    user_id: str,
    email: dict,
    label: Literal["ignore", "respond", "notify"],
) -> None:
    """Write a trusted human correction as one episodic example."""
    store.put(
        episodic_namespace(user_id),
        str(uuid.uuid4()),
        {
            "email": email,
            "label": label,
            "source": "human_correction",
        },
    )


def format_episodes(items: list) -> str:
    if not items:
        return "No relevant corrected examples are available."
    blocks = ["Relevant human-corrected triage examples:"]
    for item in items:
        email = item.value["email"]
        blocks.append(
            "\n".join(
                [
                    f"Subject: {email['subject']}",
                    f"From: {email['author']}",
                    f"Content: {email['email_thread'][:350]}",
                    f"Correct classification: {item.value['label'].upper()}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Simulate drafting an email without sending it."""
    return (
        "SIMULATION ONLY: no email was sent. "
        f"Drafted email to {to} with subject {subject!r}. Content: {content}"
    )


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: str,
) -> str:
    """Simulate proposing a meeting without creating an event."""
    return (
        "SIMULATION ONLY: no event was created. "
        f"Proposed {duration_minutes}-minute meeting {subject!r} on "
        f"{preferred_day} with {len(attendees)} attendees."
    )


@tool
def check_calendar_availability(day: str) -> str:
    """Return simulated availability."""
    return f"SIMULATION ONLY: available on {day} at 9:00 AM, 2:00 PM, or 4:00 PM"


def triage_router(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
) -> Command[Literal["response_agent", "__end__"]]:
    """Load procedures, retrieve episodes, rebuild the prompt, and route."""
    user_id = config["configurable"]["langgraph_user_id"]
    initialize_procedures(user_id)
    email = state["email_input"]

    episodes = store.search(
        episodic_namespace(user_id),
        query=str({"email": email}),
        limit=3,
    )
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=read_procedure(user_id, "triage_ignore"),
        triage_notify=read_procedure(user_id, "triage_notify"),
        triage_email=read_procedure(user_id, "triage_respond"),
        examples=(
            f"{format_episodes(episodes)}\n\n"
            "Treat relevant human corrections as strong evidence, but do not "
            "copy a label when the new email is materially different."
        ),
    )
    user_prompt = triage_user_prompt.format(
        author=email["author"],
        to=email["to"],
        subject=email["subject"],
        email_thread=email["email_thread"],
    )
    result = router_model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    print(
        f"Classification: {result.classification.upper()} "
        f"(retrieved episodes: {len(episodes)})"
    )
    print(f"Reasoning: {result.reasoning}")

    if result.classification != "respond":
        return Command(goto=END)

    return Command(
        goto="response_agent",
        update={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Handle this email with simulated tools. Search semantic "
                        "memory first when it may be a follow-up, then save useful "
                        "stable context. Do not invent progress or real actions.\n\n"
                        f"{email}"
                    ),
                }
            ]
        },
    )


def response_agent_node(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """Rebuild the response-agent prompt from the latest stored procedure."""
    user_id = config["configurable"]["langgraph_user_id"]
    initialize_procedures(user_id)
    instructions = read_procedure(user_id, "agent_instructions")
    system_prompt = agent_system_prompt_memory.format(
        instructions=instructions,
        profile=profile,
        **profile,
    )
    agent = create_agent(
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
    input_messages = state["messages"]
    result = agent.invoke({"messages": input_messages}, config=config)
    return {"messages": result["messages"][len(input_messages) :]}


def update_procedures(
    user_id: str,
    previous_interaction: list,
    feedback: str,
) -> list[str]:
    """Optimize stored procedures from feedback, then save changed prompts."""
    initialize_procedures(user_id)
    prompts = [
        {
            "name": name,
            "prompt": read_procedure(user_id, key),
            "update_instructions": "Keep the instruction short and specific.",
            "when_to_update": when_to_update,
        }
        for name, (key, when_to_update) in PROMPT_SPECS.items()
    ]
    optimized = procedure_optimizer.invoke(
        {
            "trajectories": [(previous_interaction, feedback)],
            "prompts": prompts,
        }
    )

    changed = []
    namespace = procedural_namespace(user_id)
    for old_prompt, new_prompt in zip(prompts, optimized):
        if new_prompt["prompt"] == old_prompt["prompt"]:
            continue
        name = old_prompt["name"]
        key = PROMPT_SPECS[name][0]
        store.put(
            namespace,
            key,
            {
                "prompt": new_prompt["prompt"],
                "source": "optimized_from_user_feedback",
            },
            index=False,
        )
        changed.append(name)
    return changed


def build_email_agent():
    builder = StateGraph(State)
    builder.add_node("triage_router", triage_router)
    builder.add_node("response_agent", response_agent_node)
    builder.add_edge(START, "triage_router")
    builder.add_edge("response_agent", END)
    return builder.compile(checkpointer=MemorySaver(), store=store)


def run_email(agent, user_id: str, thread_id: str, email: dict, title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    return agent.invoke(
        {"email_input": email, "messages": []},
        config={
            "configurable": {
                "langgraph_user_id": user_id,
                "thread_id": thread_id,
            }
        },
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    user_id = "john"
    initialize_procedures(user_id)
    agent = build_email_agent()

    # EPISODIC WRITE: a human correction becomes a trusted example.
    corrected_vendor_email = {
        "author": "Tom <tom@vendor.example>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about documentation",
        "email_thread": "Would you like to buy our premium API documentation bundle?",
    }
    save_episodic_correction(user_id, corrected_vendor_email, "ignore")
    similar_vendor_email = {
        "author": "Jim <jim@vendor.example>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Question about API docs",
        "email_thread": "Can I interest you in purchasing our documentation package?",
    }
    run_email(
        agent,
        user_id,
        "vendor-pitch",
        similar_vendor_email,
        "EPISODIC MEMORY — RETRIEVE A HUMAN-CORRECTED EXAMPLE",
    )

    # PROCEDURAL WRITE: feedback updates a stored rule by exact key.
    feedback_email = {
        "author": "Sarah <sarah@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Deployment complete",
        "email_thread": "Staging deployment passed. No action required; FYI only.",
    }
    feedback_trajectory = [
        HumanMessage(content=f"Classify this email: {feedback_email}"),
        AIMessage(content="Classification: notify"),
    ]
    feedback = (
        "Ignore routine deployment FYIs when they explicitly say no action is required."
    )
    print(f"\nPROCEDURAL FEEDBACK: {feedback}")
    print(
        "Updated procedures:",
        update_procedures(user_id, feedback_trajectory, feedback),
    )
    run_email(
        agent,
        user_id,
        "deployment-fyi",
        feedback_email,
        "PROCEDURAL MEMORY — REBUILD PROMPT FROM UPDATED RULES",
    )

    # SEMANTIC WRITE: the response agent stores Alice's request.
    initial_email = {
        "author": "Alice <alice@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Missing authentication docs",
        "email_thread": (
            "Are /auth/refresh and /auth/validate intentionally missing from "
            "the authentication-service documentation?"
        ),
    }
    run_email(
        agent,
        user_id,
        "alice-initial",
        initial_email,
        "SEMANTIC MEMORY — HANDLE AND STORE THE INITIAL REQUEST",
    )

    # SEMANTIC READ across a different thread_id.
    follow_up = {
        "author": "Alice <alice@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Following up",
        "email_thread": "Any update on my previous question?",
    }
    run_email(
        agent,
        user_id,
        "alice-follow-up",
        follow_up,
        "SEMANTIC MEMORY — SEARCH FROM A DIFFERENT THREAD",
    )

    print("\nFINAL SHARED STORE NAMESPACES")
    for namespace in store.list_namespaces():
        print(f"- {tuple(namespace)}")


if __name__ == "__main__":
    main()
