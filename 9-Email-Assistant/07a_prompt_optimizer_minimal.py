# Minimal procedural-memory example: optimize one stored prompt from feedback.
#
# This intentionally omits the email graph, tools, embeddings, and routing so
# the LangMem prompt-update cycle is easy to see.
#
# Run from the repository root:
#   python "9-Email-Assistant/07a_prompt_optimizer_minimal.py"

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from langmem import create_multi_prompt_optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

USER_ID = "john"
NAMESPACE = ("email_assistant", USER_ID, "procedures")
PROMPT_KEY = "triage_notify"

DEFAULT_RULE = "Notify John about all project status updates."

SYSTEM_TEMPLATE = """You are John's email triage assistant.

Emails that should notify John:
{notify_rule}

Classify the current email as IGNORE, NOTIFY, or RESPOND.
"""


def save_initial_rule(store: InMemoryStore) -> None:
    """Create the default only when no learned rule already exists."""
    if store.get(NAMESPACE, PROMPT_KEY) is None:
        store.put(
            NAMESPACE,
            PROMPT_KEY,
            {"prompt": DEFAULT_RULE, "source": "default"},
        )


def read_rule(store: InMemoryStore) -> str:
    """Read the exact stored rule; no embedding search is involved."""
    item = store.get(NAMESPACE, PROMPT_KEY)
    if item is None:
        raise KeyError(f"Missing stored prompt: {PROMPT_KEY}")
    return item.value["prompt"]


def rebuild_system_prompt(store: InMemoryStore) -> str:
    """Build a fresh system prompt from the latest stored value."""
    return SYSTEM_TEMPLATE.format(notify_rule=read_rule(store))


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    # Store remembers the optimized prompt. The optimizer itself does not
    # provide persistence.
    store = InMemoryStore()
    save_initial_rule(store)

    old_rule = read_rule(store)
    print("\n1. STORED RULE BEFORE FEEDBACK")
    print(old_rule)

    # This trajectory shows the behavior that caused the feedback.
    deployment_email = (
        "Staging deployment completed successfully. No action is required; "
        "this is only an FYI."
    )
    previous_interaction = [
        HumanMessage(content=f"Classify this email: {deployment_email}"),
        AIMessage(content="Classification: NOTIFY"),
    ]
    user_feedback = (
        "Ignore routine deployment FYIs when they explicitly say no action "
        "is required."
    )

    # This creates a separate GPT-powered optimizer call. It is not the router
    # call that classified the email.
    optimizer = create_multi_prompt_optimizer(
        "openai:gpt-4o-mini",
        kind="prompt_memory",
    )

    # A list is used because create_multi_prompt_optimizer can update several
    # named prompt sections. This minimal example supplies only one.
    prompts = [
        {
            "name": "triage-notify",
            "prompt": old_rule,
            "update_instructions": "Keep the rule short and specific.",
            "when_to_update": (
                "Update when feedback changes which emails should notify John."
            ),
        }
    ]
    optimized = optimizer.invoke(
        {
            "trajectories": [(previous_interaction, user_feedback)],
            "prompts": prompts,
        }
    )
    proposed_rule = optimized[0]["prompt"]

    print("\n2. USER FEEDBACK")
    print(user_feedback)
    print("\n3. OPTIMIZER'S PROPOSED RULE")
    print(proposed_rule)

    # In production, show sensitive policy changes to the user before this
    # write. The tutorial treats this proposal as approved.
    store.put(
        NAMESPACE,
        PROMPT_KEY,
        {
            "prompt": proposed_rule,
            "source": "optimized_from_user_feedback",
        },
    )

    print("\n4. EXACT VALUE READ BACK FROM THE STORE")
    print(read_rule(store))

    print("\n5. SYSTEM PROMPT REBUILT FOR THE NEXT EMAIL")
    print(rebuild_system_prompt(store))


if __name__ == "__main__":
    main()
