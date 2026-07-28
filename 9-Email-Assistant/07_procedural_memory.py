# Seventh Email Assistant lesson: update stored triage procedures from feedback.
#
# This example keeps triage rules in a per-user Store namespace. A separate
# LangMem prompt optimizer uses feedback to revise the appropriate rule, then
# the email router loads the new rule on its next invocation.
#
# Procedural memory uses exact Store get/put operations here, not embedding
# similarity search.
#
# Run from the repository root:
#   python "9-Email-Assistant/07_procedural_memory.py"

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from langmem import create_multi_prompt_optimizer
from pydantic import BaseModel, Field

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

DEFAULT_PROCEDURES = {
    "triage_ignore": "Marketing newsletters, spam, and unsolicited sales messages",
    "triage_notify": (
        "Team member absences, build notifications, and project status updates"
    ),
    "triage_respond": (
        "Direct questions, meeting requests, and critical bug reports"
    ),
    "agent_instructions": (
        "Use email and calendar tools when appropriate. Keep drafts concise."
    ),
}

PROMPT_SPECS = {
    "agent-instructions": {
        "key": "agent_instructions",
        "when_to_update": (
            "Update when feedback changes how emails should be written or how "
            "calendar tools should be used."
        ),
    },
    "triage-ignore": {
        "key": "triage_ignore",
        "when_to_update": (
            "Update when feedback changes which emails should be ignored."
        ),
    },
    "triage-notify": {
        "key": "triage_notify",
        "when_to_update": (
            "Update when feedback changes which emails should trigger a notification."
        ),
    },
    "triage-respond": {
        "key": "triage_respond",
        "when_to_update": (
            "Update when feedback changes which emails require a response."
        ),
    },
}


class Router(BaseModel):
    """Classify an unread email using the user's stored procedures."""

    reasoning: str = Field(description="A concise reason for the classification.")
    classification: Literal["ignore", "respond", "notify"]


# No embedding index is configured. Procedural prompts have stable keys and are
# loaded with exact get(), then replaced with put() after approved feedback.
store = InMemoryStore()
router_model = init_chat_model("openai:gpt-4o-mini").with_structured_output(Router)
procedure_optimizer = create_multi_prompt_optimizer(
    "openai:gpt-4o-mini",
    kind="prompt_memory",
)


def procedures_namespace(user_id: str) -> tuple[str, str, str]:
    return ("email_assistant", user_id, "procedures")


def initialize_procedures(user_id: str) -> None:
    """Create this user's defaults without overwriting learned instructions."""
    namespace = procedures_namespace(user_id)
    for key, prompt in DEFAULT_PROCEDURES.items():
        if store.get(namespace, key) is None:
            store.put(namespace, key, {"prompt": prompt})


def read_procedure(user_id: str, key: str) -> str:
    """Load one instruction by exact key."""
    item = store.get(procedures_namespace(user_id), key)
    if item is None:
        raise KeyError(f"Missing procedure {key!r} for user {user_id!r}")
    return item.value["prompt"]


def classify_email(user_id: str, email: dict) -> Router:
    """Build the system prompt from stored procedures and classify one email."""
    initialize_procedures(user_id)
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=read_procedure(user_id, "triage_ignore"),
        triage_notify=read_procedure(user_id, "triage_notify"),
        triage_email=read_procedure(user_id, "triage_respond"),
        examples="No episodic examples are used in this focused lesson.",
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
    print(f"Classification: {result.classification.upper()}")
    print(f"Reasoning: {result.reasoning}")
    return result


def update_procedures(
    user_id: str,
    email: dict,
    previous_result: Router,
    feedback: str,
) -> list[str]:
    """Use an LLM optimizer to revise and save the relevant stored prompts."""
    initialize_procedures(user_id)
    prompts = []
    for name, spec in PROMPT_SPECS.items():
        prompts.append(
            {
                "name": name,
                "prompt": read_procedure(user_id, spec["key"]),
                "update_instructions": "Keep instructions short and specific.",
                "when_to_update": spec["when_to_update"],
            }
        )

    trajectory = [
        HumanMessage(content=f"Classify this email:\n{email}"),
        AIMessage(
            content=(
                f"Classification: {previous_result.classification}. "
                f"Reasoning: {previous_result.reasoning}"
            )
        ),
    ]
    optimized = procedure_optimizer.invoke(
        {
            "trajectories": [(trajectory, feedback)],
            "prompts": prompts,
        }
    )

    updated_names = []
    namespace = procedures_namespace(user_id)
    for old_prompt, new_prompt in zip(prompts, optimized):
        if new_prompt["prompt"] == old_prompt["prompt"]:
            continue
        name = old_prompt["name"]
        key = PROMPT_SPECS[name]["key"]
        store.put(
            namespace,
            key,
            {
                "prompt": new_prompt["prompt"],
                "source": "optimized_from_user_feedback",
            },
        )
        updated_names.append(name)
    return updated_names


def print_procedures(user_id: str) -> None:
    print("\nSTORED PROCEDURAL MEMORY")
    for name, spec in PROMPT_SPECS.items():
        print(f"- {name}: {read_procedure(user_id, spec['key'])}")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    user_id = "john"
    deployment_fyi = {
        "author": "Sarah Chen <sarah.chen@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Authentication endpoints deployed to staging",
        "email_thread": (
            "Hi John—authentication changes are deployed to staging and all "
            "tests pass. No action is required; this is only an FYI."
        ),
    }

    print("\n" + "=" * 70)
    print("BEFORE PROCEDURAL FEEDBACK")
    print("=" * 70)
    before = classify_email(user_id, deployment_fyi)
    print_procedures(user_id)

    feedback = (
        "For my workflow, ignore routine deployment FYIs when they explicitly "
        "say that no action is required."
    )
    print(f"\nUSER FEEDBACK: {feedback}")
    updated_names = update_procedures(
        user_id,
        deployment_fyi,
        before,
        feedback,
    )
    print(f"Updated prompts: {updated_names or ['none']}")
    print_procedures(user_id)

    print("\n" + "=" * 70)
    print("AFTER PROCEDURAL FEEDBACK")
    print("=" * 70)
    classify_email(user_id, deployment_fyi)

    print("\nDIFFERENT-USER ISOLATION")
    other_user = "another-user"
    initialize_procedures(other_user)
    print(
        "- John's ignore rule:",
        read_procedure(user_id, "triage_ignore"),
    )
    print(
        "- Other user's ignore rule:",
        read_procedure(other_user, "triage_ignore"),
    )


if __name__ == "__main__":
    main()
