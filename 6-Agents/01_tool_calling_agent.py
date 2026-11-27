# Run from the repository root:
#   python "6-Agents/01_tool_calling_agent.py"
#
# A more complete tool-calling agent built from prebuilt parts (book Listing 7.3:
# ToolNode runs the tools, tools_condition routes, MessagesState holds history):
# the LLM can call a live weather tool, a tip calculator, and (if a Tavily
# key is set) a web search tool, looping until it has a final answer.

import os
import sys
from pathlib import Path

import requests

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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
# 3. Node
#
# MessagesState already declares messages with the add_messages
# reducer, so every lap appends to the history instead of replacing it.
# ---------------------------------------------------------
def call_llm(state: MessagesState) -> dict:
    """Send messages to the LLM and receive a response."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------
# 4. Build The Agent Graph (book Listing 7.3)
#
# Agent loop:
# llm -> tools_condition -> tools -> llm -> ... -> END
#
# ToolNode runs whichever tool the LLM requested; tools_condition is the
# prebuilt router: "tools" if the last AIMessage has tool calls, else END.
# The number of tool iterations is not fixed ahead of time.
# ---------------------------------------------------------
builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)     # "tools" or END
builder.add_edge("tools", "llm")                           # the back edge
agent = builder.compile()


# ---------------------------------------------------------
# 5. Run It
# ---------------------------------------------------------
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "00_tool_calling_agent_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(agent, graph_image_path)

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

        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        final_message = result["messages"][-1]

        print("Final answer:", final_message.content)


if __name__ == "__main__":
    main()
