# Run from the repository root:
#   python "6-Agents/10_two_agents_with_state.py"
#
# Same two create_agent specialists as 09_two_agents_direct.py, but an outer
# StateGraph owns the handoff: START -> analyst -> writer -> END.
# Each node invokes one agent and writes one state field. invoke() returns
# the full state {request, analysis, report} — not a Python variable you
# thread by hand.
#
# Prerequisites: OPENAI_API_KEY in the repo-root .env.

import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from util import plot_graph  # noqa: E402


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


class State(TypedDict):
    request: str
    analysis: str
    report: str


def tools_executed(messages: list) -> list[str]:
    names = [m.name for m in messages if isinstance(m, ToolMessage) and m.name]
    return list(dict.fromkeys(names))


def analyst_node(state: State) -> dict:
    result = analyst.invoke(
        {"messages": [HumanMessage(state["request"])]}
    )
    print("Analyst tools:", tools_executed(result["messages"]) or "(none)")
    return {"analysis": result["messages"][-1].content}


def writer_node(state: State) -> dict:
    prompt = (
        f"Original request:\n{state['request']}\n\n"
        f"Analysis from the analyst:\n{state['analysis']}\n\n"
        "Write the final report."
    )
    result = writer.invoke({"messages": [HumanMessage(prompt)]})
    print("Writer tools:", tools_executed(result["messages"]) or "(none)")
    return {"report": result["messages"][-1].content}


builder = StateGraph(State)
builder.add_node("analyst", analyst_node)
builder.add_node("writer", writer_node)
builder.add_edge(START, "analyst")
builder.add_edge("analyst", "writer")
builder.add_edge("writer", END)
graph = builder.compile()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repo-root .env.")

    png = Path(__file__).resolve().parent / "diagrams" / "10_two_agents_with_state.png"
    png.parent.mkdir(exist_ok=True)
    plot_graph(graph, str(png), print_mermaid=False)

    result = graph.invoke(
        {
            "request": (
                "We sold 120 units at $25 each. "
                "Total costs were $1,800. Analyze the financial results."
            ),
            "analysis": "",
            "report": "",
        }
    )
    print("\n=== ANALYSIS ===")
    print(result["analysis"])
    print("\n=== FINAL REPORT ===")
    print(result["report"])


if __name__ == "__main__":
    main()
