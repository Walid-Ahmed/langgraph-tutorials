# Third Email Assistant lesson: combine triage and the tool-calling response
# agent in one LangGraph.
#
# Flow:
#   START -> triage_router -> response_agent (RESPOND only)
#                          -> END            (IGNORE or NOTIFY)
#
# All tools are simulations. This script sends no email and changes no calendar.
#
# Run from the repository root:
#   python "9-Email-Assistant/03_overall_email_agent.py"

import os
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from email_assistant.prompts import (
    agent_system_prompt,
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
        "Use the available tools when appropriate to manage John's tasks "
        "efficiently."
    ),
}


class State(TypedDict):
    # email_input is the current work item. add_messages appends response-agent
    # messages instead of overwriting them when a node returns an update.
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
    """Simulate writing and sending an email."""
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
    return (
        f"Simulated {duration_minutes}-minute meeting {subject!r} on "
        f"{preferred_day} with {len(attendees)} attendees."
    )


@tool
def check_calendar_availability(day: str) -> str:
    """Return simulated calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


# The router and response agent use the same model name but are separate calls:
# one makes a deterministic routing decision; the other can loop over tools.
llm = init_chat_model("openai:gpt-4o-mini")
llm_router = llm.with_structured_output(Router)

response_agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[write_email, schedule_meeting, check_calendar_availability],
    system_prompt=agent_system_prompt.format(
        instructions=prompt_instructions["agent_instructions"],
        **profile,
    ),
)


def triage_router(
    state: State,
) -> Command[Literal["response_agent", "__end__"]]:
    """Classify the email and choose the next graph node."""
    email_input = state["email_input"]

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples="No episodic examples are used yet.",
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
        # Command combines a state update with the routing decision. The new
        # message becomes the response agent's task.
        return Command(
            goto="response_agent",
            update={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Respond to the incoming email below. Any tool in "
                            "this tutorial is simulated and causes no external "
                            "action.\n\n"
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


def build_graph():
    # The router chooses the destination at runtime, so only START needs a
    # static edge. Command(goto=...) supplies the remaining transitions.
    builder = StateGraph(State)
    builder.add_node("triage_router", triage_router)
    builder.add_node("response_agent", response_agent)
    builder.add_edge(START, "triage_router")
    return builder.compile()


def render_graph(email_agent) -> None:
    """Print Mermaid source and try to save a PNG diagram."""
    graph_view = email_agent.get_graph(xray=True)
    print("\nMERMAID GRAPH")
    print(graph_view.draw_mermaid())

    output_path = (
        REPO_ROOT / "9-Email-Assistant/diagrams/overall_email_agent_graph.png"
    )
    try:
        output_path.write_bytes(graph_view.draw_mermaid_png())
        print("\nPLOTTED GRAPH FILE")
        print(f"Saved to: {output_path}")
        print("Open this PNG file to view the rendered LangGraph.")
    except Exception as exc:
        # Diagram rendering uses an external renderer by default. The tutorial
        # can still run when that service is unavailable.
        print("\nPLOTTED GRAPH FILE")
        print(f"PNG was not saved to: {output_path}")
        print(f"Rendering failed: {exc}")


def run_email(email_agent, email_input: dict, example_name: str) -> None:
    """Run one email through the graph and display any response-agent trace."""
    print("\n" + "=" * 70)
    print(f"TEST EMAIL: {example_name}")
    print(f"Subject: {email_input['subject']}")
    print("=" * 70)

    result = email_agent.invoke(
        {
            "email_input": email_input,
            "messages": [],
        }
    )

    messages = result.get("messages", [])
    if messages:
        print("\nRESPONSE-AGENT MESSAGE TRACE")
        for message in messages:
            message.pretty_print()
    else:
        print("\nNo response-agent messages were produced for this route.")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    email_agent = build_graph()
    render_graph(email_agent)

    marketing_email = {
        "author": "Marketing Team <marketing@amazingdeals.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": (
            "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥"
        ),
        "email_thread": (
            "Dear Valued Developer,\n\n"
            "For a limited time, get 80% off our Premium Developer Suite. "
            "This offer expires in 24 hours.\n\n"
            "Click here to claim your discount: "
            "https://amazingdeals.com/special-offer\n\n"
            "Best regards,\nMarketing Team\n\n"
            "To unsubscribe, click here."
        ),
    }
    run_email(
        email_agent,
        marketing_email,
        "Marketing offer (expected: IGNORE)",
    )

    documentation_email = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": (
            "Hi John,\n\nThe /auth/refresh and /auth/validate endpoints "
            "seem to be missing from the authentication-service specs. "
            "Was that intentional, or should we update the docs?\n\n"
            "Thanks!\nAlice"
        ),
    }
    run_email(
        email_agent,
        documentation_email,
        "Direct question from a teammate (expected: RESPOND)",
    )


if __name__ == "__main__":
    main()
