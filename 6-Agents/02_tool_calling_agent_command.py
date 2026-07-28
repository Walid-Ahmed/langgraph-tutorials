# Run from the repository root:
#   python "6-Agents/02_tool_calling_agent_command.py"
#
# A low-level tool loop where the LLM node returns Command to update state and
# choose the next node. No router function or conditional edges are needed.

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

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


tools = [add, multiply]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


def call_model(
    state: MessagesState,
) -> Command[Literal["tools", "__end__"]]:
    """Call the model, append its response, and route with Command."""
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are an arithmetic assistant. Use the provided tools "
                    "for calculations."
                )
            ),
            *state["messages"],
        ]
    )

    # Command replaces both a separate router and add_conditional_edges().
    next_node = "tools" if response.tool_calls else END
    return Command(
        update={"messages": [response]},
        goto=next_node,
    )


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "llm")
    builder.add_edge("tools", "llm")
    return builder.compile()


def render_graph(graph) -> None:
    """Print Mermaid source and try to save the compiled graph as PNG."""
    graph_view = graph.get_graph()
    print("\nMERMAID GRAPH")
    print(graph_view.draw_mermaid())

    output_path = (
        REPO_ROOT / "6-Agents/diagrams/02_tool_calling_agent_command.png"
    )
    try:
        output_path.write_bytes(graph_view.draw_mermaid_png())
        print(f"\nGraph saved to: {output_path}")
    except Exception as exc:
        print(f"\nPNG rendering skipped: {exc}")


def save_message_trace(messages: list) -> Path:
    """Save the complete interaction trace and return its log-file path."""
    log_path = REPO_ROOT / "6-Agents/logs/02_command_agent_messages.log"
    log_path.parent.mkdir(exist_ok=True)
    trace = "\n\n".join(message.pretty_repr() for message in messages)
    log_path.write_text(f"{trace}\n", encoding="utf-8")
    return log_path


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    graph = build_graph()
    render_graph(graph)
    result = graph.invoke(
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
