# Run from the repository root:
#   python "6-Agents/09_two_agents_direct.py"
#
# Two create_agent graphs, no outer StateGraph. Agent 1 (analyst) runs first.
# Python copies its last message into Agent 2's (writer) prompt. That is the
# whole "connection": a string in a variable, not a graph edge.
#
# Compare with 10_two_agents_with_state.py, which does the same handoff through
# LangGraph state (request / analysis / report).
#
# Prerequisites: OPENAI_API_KEY in the repo-root .env.

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def word_count(text: str) -> int:
    """Count the words in text."""
    return len(text.split())


REQUEST = (
    "We sold 120 units at $25 each. "
    "Total costs were $1,800. Analyze the financial results."
)


def tools_executed(messages: list) -> list[str]:
    """Terminal helper: names on ToolMessage (tools that actually ran)."""
    names = [m.name for m in messages if isinstance(m, ToolMessage) and m.name]
    return list(dict.fromkeys(names))


def last_text(messages: list) -> str:
    return str(messages[-1].content) if messages else ""


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repo-root .env.")

    analyst = create_agent(
        model="openai:gpt-4o-mini",
        tools=[multiply, subtract],
        system_prompt=(
            "You are a financial analyst. Calculate revenue and profit. "
            "Always use the provided tools for arithmetic."
        ),
    )
    writer = create_agent(
        model="openai:gpt-4o-mini",
        tools=[word_count],
        system_prompt=(
            "You are a report writer. Turn the supplied analysis into a clear "
            "report of at most 80 words. Use word_count to verify the length."
        ),
    )

    result1 = analyst.invoke({"messages": [HumanMessage(REQUEST)]})
    analysis = last_text(result1["messages"])
    print("=== AGENT 1: ANALYST ===")
    print("Tools:", tools_executed(result1["messages"]) or "(none)")
    print(analysis)

    result2 = writer.invoke(
        {
            "messages": [
                HumanMessage(
                    "The first agent produced this analysis:\n\n"
                    f"{analysis}\n\nWrite the final report."
                )
            ]
        }
    )
    print("\n=== AGENT 2: WRITER ===")
    print("Tools:", tools_executed(result2["messages"]) or "(none)")
    print(last_text(result2["messages"]))


if __name__ == "__main__":
    main()
