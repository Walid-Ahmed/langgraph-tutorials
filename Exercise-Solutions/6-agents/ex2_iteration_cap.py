# Exercise 2 — Cap the agent loop at 5 iterations
#
# Adds an iteration_count field to the state.
# call_llm increments it on every pass.
# should_use_tools routes to END if iteration_count >= 5,
# even when the model still wants to call tools.

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
def always_needs_more(query: str) -> str:
    """A tool that always asks the model to dig deeper (for testing the cap)."""
    return f"Interesting point about '{query}'. Please search for more details."


tools = [always_needs_more]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int


def call_llm(state: AgentState) -> dict:
    iteration = state["iteration_count"] + 1
    print(f"Iteration {iteration}")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "iteration_count": iteration}


tool_node = ToolNode(tools)


def should_use_tools(state: AgentState) -> str:
    if state["iteration_count"] >= 5:
        print("Iteration cap reached — stopping.")
        return END
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
    result = app.invoke({
        "messages": [HumanMessage(content="Tell me everything about AI agents.")],
        "iteration_count": 0,
    })
    print(f"\nFinal iteration_count: {result['iteration_count']}")
    print(f"Final answer: {result['messages'][-1].content[:300]}")


if __name__ == "__main__":
    main()
