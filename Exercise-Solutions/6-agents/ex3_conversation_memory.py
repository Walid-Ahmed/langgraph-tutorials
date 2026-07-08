# Exercise 3 — Multi-turn agent with carry-over history
#
# After the first graph run, the full message history is captured.
# The second run passes that history plus a follow-up question.
# The agent should recall context from the first turn.

import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


@tool
def calculate_tip(bill_amount: float, tip_percentage: float) -> float:
    """Calculate tip amount based on bill and percentage."""
    return round(bill_amount * (tip_percentage / 100), 2)


tools = [calculate_tip]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_llm(state: AgentState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools)


def should_use_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_use_tools, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")
app = graph.compile()
plot_graph(app)


def main() -> None:
    # Turn 1
    print("=== Turn 1 ===")
    turn1 = app.invoke({
        "messages": [HumanMessage(content="Calculate a 20% tip on a $50 bill.")]
    })
    print(f"Agent: {turn1['messages'][-1].content}")

    # Turn 2 — pass the full history from turn 1 plus a follow-up
    print("\n=== Turn 2 (with memory) ===")
    turn2 = app.invoke({
        "messages": turn1["messages"] + [
            HumanMessage(content="What was the bill amount I asked about?")
        ]
    })
    print(f"Agent: {turn2['messages'][-1].content}")
    # The agent should mention $50 because it sees the full prior history.


if __name__ == "__main__":
    main()
