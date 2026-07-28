# Second Email Assistant lesson: give the main agent communication and calendar
# tools. The tools are simulations; they do not send email or change a calendar.
#
# Run from the repository root:
#   python "9-Email-Assistant/02_main_agent_tools.py"

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

from email_assistant.prompts import agent_system_prompt

load_dotenv()


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
        "efficiently."
    )
}


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Simulate writing and sending an email."""
    # @tool exposes this signature and docstring as a schema the model can call.
    # A production implementation would require approval and call an email API.
    return (
        f"Simulated email to {to} with subject {subject!r}. "
        f"Content: {content}"
    )


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: str,
) -> str:
    """Simulate scheduling a calendar meeting."""
    # A production implementation would check conflicts and call a calendar API.
    return (
        f"Simulated {duration_minutes}-minute meeting {subject!r} on "
        f"{preferred_day} with {len(attendees)} attendees."
    )


@tool
def check_calendar_availability(day: str) -> str:
    """Return simulated calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


def create_system_prompt() -> str:
    """Build the instructions supplied to the main agent."""
    return agent_system_prompt.format(
        instructions=prompt_instructions["agent_instructions"],
        **profile,
    )


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    # create_agent builds the model -> tool -> model loop. The model decides
    # whether a tool is needed; these Python functions perform the actual call.
    tools = [write_email, schedule_meeting, check_calendar_availability]
    agent = create_agent(
        "openai:gpt-4o-mini",
        tools=tools,
        system_prompt=create_system_prompt(),
    )

    # The returned message list includes the user request, any tool calls and
    # tool results, and the final assistant answer.
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is my availability for Tuesday?",
                }
            ]
        }
    )
    response["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
