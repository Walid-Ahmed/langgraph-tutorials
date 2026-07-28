# Smallest prompt-optimizer example: revise one Python string without a Store.
#
# This demonstrates prompt optimization, not long-term procedural memory.
# The revised prompt disappears when this Python process ends.
#
# Run from the repository root:
#   python "9-Email-Assistant/07b_prompt_optimizer_no_store.py"

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langmem import create_multi_prompt_optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

SYSTEM_TEMPLATE = """You are John's email triage assistant.

Emails that should notify John:
{notify_rule}

Classify the current email as IGNORE, NOTIFY, or RESPOND.
"""


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    # The current procedure is only a Python string. Nothing is persisted.
    notify_rule = "Notify John about all project status updates."

    previous_interaction = [
        HumanMessage(
            content=(
                "Classify this email: Staging deployment completed. "
                "No action is required; this is only an FYI."
            )
        ),
        AIMessage(content="Classification: NOTIFY"),
    ]
    user_feedback = (
        "Ignore routine deployment FYIs when they explicitly say no action "
        "is required."
    )

    # The optimizer is a separate GPT-4o-mini call that proposes replacement
    # text for the supplied prompt string.
    optimizer = create_multi_prompt_optimizer(
        "openai:gpt-4o-mini",
        kind="prompt_memory",
    )
    optimized = optimizer.invoke(
        {
            "trajectories": [(previous_interaction, user_feedback)],
            "prompts": [
                {
                    "name": "triage-notify",
                    "prompt": notify_rule,
                    "update_instructions": "Keep the rule short and specific.",
                    "when_to_update": (
                        "Update when feedback changes which emails notify John."
                    ),
                }
            ],
        }
    )

    # Replace the local variable with the proposal. There is no Store write.
    optimized_rule = optimized[0]["prompt"]

    print("\n1. ORIGINAL PYTHON STRING")
    print(notify_rule)
    print("\n2. USER FEEDBACK")
    print(user_feedback)
    print("\n3. OPTIMIZED PYTHON STRING")
    print(optimized_rule)
    print("\n4. SYSTEM PROMPT REBUILT IN THIS PROCESS")
    print(SYSTEM_TEMPLATE.format(notify_rule=optimized_rule))
    print(
        "\nNo Store is used. Restarting this script restores the original "
        "hard-coded rule."
    )


if __name__ == "__main__":
    main()
