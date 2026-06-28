import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

try:
    from langchain_tavily import TavilySearch
except ImportError:
    TavilySearch = None


load_dotenv()


# ---------------------------------------------------------
# 1. Tools
#
# An augmented LLM can use tools instead of only answering
# from its internal model knowledge.
# ---------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    print("tool weather is called")

    weather_data = {
        "New York": "Sunny, 72 F",
        "London": "Cloudy, 15 C",
        "Tokyo": "Rainy, 20 C",
    }

    return weather_data.get(city, "Weather data not available")


@tool
def calculate_tip(bill_amount: float, tip_percentage: float) -> float:
    """Calculate tip amount based on bill and percentage."""
    print("tool tip_calc is called")

    return round(bill_amount * (tip_percentage / 100), 2)


# Start with local tools that do not need external API keys.
tools = [get_weather, calculate_tip]


# Tavily search is optional because it requires a TAVILY_API_KEY.
# If the package and API key are available, the LLM can also search the web.
search_enabled = TavilySearch is not None and bool(os.getenv("TAVILY_API_KEY"))

if search_enabled:
    tavily_search = TavilySearch(max_results=3, search_depth="basic")
    tools.append(tavily_search)


# ---------------------------------------------------------
# 2. LLM With Tools Bound
#
# bind_tools() tells the model which tools it is allowed to call.
# ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------
# 3. State Definition
#
# messages stores the conversation history.
# add_messages appends new messages instead of replacing history.
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------
# 4. Nodes
# ---------------------------------------------------------
def call_llm(state: AgentState) -> dict:
    """Send messages to the LLM and receive a response."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ToolNode automatically runs whichever tool the LLM requested.
tool_node = ToolNode(tools)


# ---------------------------------------------------------
# 5. Routing Logic
#
# If the LLM response contains tool calls, route to the tools node.
# Otherwise, the graph can end.
# ---------------------------------------------------------
def should_use_tools(state: AgentState) -> str:
    """Route to tools if the last LLM message requested tool calls."""
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ---------------------------------------------------------
# 6. Build The Graph
#
# Flow:
# llm -> tools -> llm -> END
#
# The graph loops back to the LLM after tools run so the model can
# turn tool results into a final answer.
# ---------------------------------------------------------
graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("llm")

graph_builder.add_conditional_edges(
    "llm",
    should_use_tools,
    {
        "tools": "tools",
        END: END,
    },
)

graph_builder.add_edge("tools", "llm")

graph = graph_builder.compile()


# ---------------------------------------------------------
# 7. Run It
# ---------------------------------------------------------
def save_graph_png() -> None:
    """Save the graph image when Mermaid rendering is available."""
    try:
        with open("graph.png", "wb") as file:
            file.write(graph.get_graph().draw_mermaid_png())
        print("Graph saved to graph.png")
    except Exception as error:
        print("Could not save graph.png:", error)
        print("Mermaid graph:")
        print(graph.get_graph().draw_mermaid())


def main() -> None:
    save_graph_png()

    prompts = [
        "What's the weather in London?",
        "Calculate a 20% tip on a $50 bill",
    ]

    if search_enabled:
        prompts.append("Search for the latest news about AI agents")

    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print("=" * 60)

        result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
        final_message = result["messages"][-1]

        print("Final answer:", final_message.content)


if __name__ == "__main__":
    main()
