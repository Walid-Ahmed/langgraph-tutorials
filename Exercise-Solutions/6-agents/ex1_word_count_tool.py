# Exercise 1 — Add a word_count tool
#
# Adds a word_count(text) tool alongside get_weather and calculate_tip.
# Ask the agent a question that requires word counting and verify it
# calls the tool rather than guessing.

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


@tool
def word_count(text: str) -> int:
    """Count the number of words in a string."""
    print(f"word_count called with: '{text}'")
    return len(text.split())


tools = [calculate_tip, word_count]
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
    prompts = [
        "How many words are in the phrase 'the quick brown fox'?",
        "Calculate a 15% tip on a $80 bill.",
    ]
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        result = app.invoke({"messages": [HumanMessage(content=prompt)]})
        print(f"Answer: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
