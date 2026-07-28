# First Email Assistant lesson: classify one incoming email.
#
# This lesson deliberately does not build a LangGraph yet. It tests the router
# that a later graph will use.
#
# Memory concepts visible at this stage:
# - profile: semantic facts about the user
# - prompt_instructions: procedural rules for triage
# - examples=None: episodic few-shot memory is not active yet
#
# Run from the repository root:
#   python "9-Email-Assistant/01_triage.py"

import os
import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from email_assistant.prompts import triage_system_prompt, triage_user_prompt

load_dotenv()

TUTORIAL_DIR = Path(__file__).resolve().parent
LOG_DIR = TUTORIAL_DIR / "logs"
PROMPT_LOG_PATH = LOG_DIR / "triage_prompts.log"


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
        "Use the available tools when appropriate to manage John's tasks "
        "efficiently."
    ),
}

email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
Hi John,

I was reviewing the API documentation for the new authentication service and
noticed a few endpoints seem to be missing from the specs. Could you help
clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice
""".strip(),
}


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


def log_prompts(system_prompt: str, user_prompt: str) -> None:
    """Write the exact prompts sent to the model to a local log file."""
    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger("email_assistant.triage.prompts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(
        PROMPT_LOG_PATH,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info("=== SYSTEM PROMPT ===\n%s", system_prompt.strip())
    logger.info("\n=== USER PROMPT ===\n%s", user_prompt.strip())
    file_handler.close()
    logger.removeHandler(file_handler)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    llm = init_chat_model("openai:gpt-4o-mini")
    llm_router = llm.with_structured_output(Router)

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        examples="No examples are used in this first lesson.",
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
    )
    user_prompt = triage_user_prompt.format(
        author=email["from"],
        to=email["to"],
        subject=email["subject"],
        email_thread=email["body"],
    )

    log_prompts(system_prompt, user_prompt)
    print(f"Prompts logged to: {PROMPT_LOG_PATH}")

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    print("\nEmail triage result")
    print(f"Classification: {result.classification}")
    print(f"Reasoning: {result.reasoning}")


if __name__ == "__main__":
    main()
