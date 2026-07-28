# Sixth Email Assistant lesson: improve triage with episodic memory.
#
# A user correction is saved as a labeled email example. A later, similar
# email is found with semantic search and inserted into the triage prompt as a
# few-shot example. This changes classification without changing the hard-coded
# triage rules.
#
# Run from the repository root:
#   python "9-Email-Assistant/06_episodic_memory.py"

import os
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from email_assistant.prompts import triage_system_prompt, triage_user_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": (
        "Senior software engineer leading a team of 5 developers"
    ),
}

triage_rules = {
    "ignore": "Marketing newsletters, spam, and unsolicited sales messages",
    "notify": "Team member absences, build notifications, and status updates",
    "respond": "Direct questions, meeting requests, and critical bug reports",
}


class State(TypedDict):
    email_input: dict
    classification: NotRequired[Literal["ignore", "respond", "notify"]]
    reasoning: NotRequired[str]


class Router(BaseModel):
    """Classify an unread email."""

    reasoning: str = Field(description="A concise reason for the classification.")
    classification: Literal["ignore", "respond", "notify"]


store = InMemoryStore(
    index={
        "embed": "openai:text-embedding-3-small",
        "dims": 1536,
    }
)
llm_router = init_chat_model("openai:gpt-4o-mini").with_structured_output(Router)


def examples_namespace(user_id: str) -> tuple[str, str, str]:
    """Keep each user's corrected triage examples isolated."""
    return ("email_assistant", user_id, "examples")


def format_few_shot_examples(items: list) -> str:
    """Turn retrieved episodes into compact examples for the router prompt."""
    if not items:
        return "No relevant corrected examples are available."

    examples = ["Relevant corrections from earlier email triage:"]
    for item in items:
        email = item.value["email"]
        examples.append(
            "\n".join(
                [
                    f"Subject: {email['subject']}",
                    f"From: {email['author']}",
                    f"Content: {email['email_thread'][:400]}",
                    f"Correct classification: {item.value['label'].upper()}",
                ]
            )
        )
    return "\n\n---\n\n".join(examples)


def triage_router(
    state: State,
    config: RunnableConfig,
    store: BaseStore,
) -> Command[Literal["__end__"]]:
    """Retrieve similar corrected episodes, then classify the current email."""
    user_id = config["configurable"]["langgraph_user_id"]
    email = state["email_input"]
    matches = store.search(
        examples_namespace(user_id),
        query=str({"email": email}),
        limit=3,
    )
    examples = format_few_shot_examples(matches)

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=triage_rules["ignore"],
        triage_notify=triage_rules["notify"],
        triage_email=triage_rules["respond"],
        examples=(
            f"{examples}\n\n"
            "Treat retrieved corrections as strong evidence for how John wants "
            "similar emails handled. Do not copy a label when the new email is "
            "materially different."
        ),
    )
    user_prompt = triage_user_prompt.format(
        author=email["author"],
        to=email["to"],
        subject=email["subject"],
        email_thread=email["email_thread"],
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    print(f"Classification: {result.classification.upper()}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Retrieved corrected examples: {len(matches)}")
    return Command(
        goto=END,
        update={
            "classification": result.classification,
            "reasoning": result.reasoning,
        },
    )


def save_triage_correction(
    user_id: str,
    email: dict,
    correct_label: Literal["ignore", "respond", "notify"],
) -> None:
    """Save one human-approved classification as an episodic memory."""
    store.put(
        examples_namespace(user_id),
        str(uuid.uuid4()),
        {
            "email": email,
            "label": correct_label,
            "source": "human_correction",
        },
    )


def build_agent():
    builder = StateGraph(State)
    builder.add_node("triage_router", triage_router)
    builder.add_edge(START, "triage_router")
    return builder.compile(store=store)


def run_triage(agent, email: dict, config: dict, title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    agent.invoke({"email_input": email}, config=config)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    user_id = "john"
    config = {"configurable": {"langgraph_user_id": user_id}}
    agent = build_agent()

    corrected_email = {
        "author": "Sarah Chen <sarah.chen@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Authentication endpoints deployed to staging",
        "email_thread": (
            "Hi John—JWT refresh rotation and login rate limiting are now on "
            "staging. All tests pass. No action is needed; just keeping you "
            "in the loop."
        ),
    }

    run_triage(
        agent,
        corrected_email,
        config,
        "BEFORE FEEDBACK — NO EPISODIC EXAMPLE",
    )

    print("\nHUMAN FEEDBACK: John wants no-action deployment FYIs ignored")
    save_triage_correction(user_id, corrected_email, "ignore")

    similar_email = {
        "author": "Jim Lee <jim.lee@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Payments endpoints deployed to staging",
        "email_thread": (
            "Hi John—the payments API changes are now deployed to staging and "
            "the test suite passes. Nothing needed from you; this is only an FYI."
        ),
    }
    run_triage(
        agent,
        similar_email,
        config,
        "AFTER FEEDBACK — SIMILAR EPISODE RETRIEVED",
    )

    other_user_config = {
        "configurable": {"langgraph_user_id": "another-user"}
    }
    run_triage(
        agent,
        similar_email,
        other_user_config,
        "DIFFERENT USER — JOHN'S CORRECTION IS ISOLATED",
    )


if __name__ == "__main__":
    main()
