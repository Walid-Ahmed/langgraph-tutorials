# A more complete tool-calling agent using LangGraph's prebuilt ToolNode:
# the LLM can call a live weather tool, a tip calculator, and (if a Tavily
# key is set) a web search tool, looping until it has a final answer.

import os
import sys
from pathlib import Path
from typing import Annotated

import requests

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

try:
    from langchain_tavily import TavilySearch
except ImportError:
    TavilySearch = None


load_dotenv()

weather_api_key = os.getenv("OPENWEATHER_API_KEY")


# ---------------------------------------------------------
# 1. Tools
#
# A tool-calling agent can use external tools instead of only
# answering from its internal model knowledge.
# ---------------------------------------------------------
@tool
def get_weather(destination_city: str) -> str:
    """Fetch live weather for a city from OpenWeatherMap."""
    print(f"get_weather called for {destination_city}")

    if not weather_api_key:
        return "Weather data not available: OPENWEATHER_API_KEY is missing"

    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": destination_city,
        "appid": weather_api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Weather API status: {response.status_code}")
        print(f"Weather API raw: {response.text[:200]}")

        if response.status_code == 200:
            data = response.json()
            description = data["weather"][0]["description"].capitalize()
            temperature = data["main"]["temp"]
            return f"{description} with {temperature}°C"

        return f"Weather data not available for {destination_city}"
    except Exception as error:
        return f"Error fetching weather: {error}"


@tool
def calculate_tip(bill_amount: float, tip_percentage: float) -> float:
    """Calculate tip amount based on bill and percentage."""
    print("tool tip_calc is called")

    return round(bill_amount * (tip_percentage / 100), 2)


# Start with tools that do not need Tavily.
# get_weather still needs OPENWEATHER_API_KEY to return live weather.
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
# 6. Build The Agent Graph
#
# Agent loop:
# llm -> should_use_tools -> tools -> llm -> ... -> END
#
# The number of tool iterations is not fixed ahead of time.
# The model decides whether to keep calling tools or stop.
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
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "00_tool_calling_agent_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(graph, graph_image_path)

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
