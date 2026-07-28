# Run from the repository root:
#   python "6-Agents/03_prebuilt_react_agent.py"
#
# A high-level ReAct-style tool agent. LangChain's create_agent builds and
# returns a compiled LangGraph, hiding the model/tool nodes and routing code.

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def build_agent():
    """Create a standard model-tools loop backed by LangGraph."""
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[add, multiply],
        system_prompt=(
            "You are an arithmetic assistant. Always use the provided tools "
            "for calculations."
        ),
    )


def render_graph(agent) -> None:
    """Expose and save the LangGraph created by the high-level helper."""
    graph_view = agent.get_graph(xray=True)
    print("\nMERMAID GRAPH")
    print(graph_view.draw_mermaid())

    output_path = REPO_ROOT / "6-Agents/diagrams/03_prebuilt_react_agent.png"
    try:
        output_path.write_bytes(graph_view.draw_mermaid_png())
        print(f"\nGraph saved to: {output_path}")
    except Exception as exc:
        print(f"\nPNG rendering skipped: {exc}")


def save_message_trace(messages: list) -> Path:
    """Save the complete interaction trace and return its log-file path."""
    log_path = REPO_ROOT / "6-Agents/logs/03_prebuilt_agent_messages.log"
    log_path.parent.mkdir(exist_ok=True)
    trace = "\n\n".join(message.pretty_repr() for message in messages)
    log_path.write_text(f"{trace}\n", encoding="utf-8")
    return log_path


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    agent = build_agent()
    render_graph(agent)
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Add 3 and 4, then multiply the result by 2."
                )
            ]
        }
    )

    print("\nMESSAGE TRACE")
    for message in result["messages"]:
        message.pretty_print()

    log_path = save_message_trace(result["messages"])
    print(f"\nInteraction log saved to: {log_path}")


if __name__ == "__main__":
    main()
