# Step 12 in 6-Agents: the other way to attach a subgraph - pass it to add_node.
#
# This script builds a tiny two-step graph (no LLM, no API key), then uses that
# compiled graph directly as a node of a parent graph. Parent and subgraph talk
# through the state keys they have in common; everything else stays private.
#
# Why this matters:
# - 11_parallel_subgraphs.py called each subgraph INSIDE a node function,
#   because a create_agent graph shares no keys with that parent.
# - When the two states DO share keys, add_node(name, compiled_graph) is
#   enough: shared keys flow in and out, and keys the parent does not declare
#   (here `scratch`) never reach it.
# - It does not use an agent; the subgraph is plain Python so the output is
#   the same on every run.
#
# Before you run this:
# 1. pip install -r requirements.txt   (no API key needed)
#
# Run it from the repository root with:
#    python "6-Agents/12_subgraph_shared_keys.py"
#
# Expected output:
#    parent result: {'question': 'Montreal this weekend?', 'weather_report': 'Report for: Montreal this weekend?'}
#    subgraph alone: {'question': 'Montreal this weekend?', 'weather_report': 'Report for: Montreal this weekend?', 'scratch': ['looked up the forecast', 'wrote the report']}

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class WeatherState(TypedDict):
    question: str                          # shared with the parent (read)
    weather_report: str                    # shared with the parent (written)
    scratch: Annotated[list[str], add]     # private: the parent has no such key


def look_up(state: WeatherState) -> dict:
    return {"scratch": ["looked up the forecast"]}


def write_up(state: WeatherState) -> dict:
    return {"scratch": ["wrote the report"],
            "weather_report": f"Report for: {state['question']}"}


weather_builder = StateGraph(WeatherState)
weather_builder.add_node("look_up", look_up)
weather_builder.add_node("write_up", write_up)
weather_builder.add_edge(START, "look_up")
weather_builder.add_edge("look_up", "write_up")
weather_builder.add_edge("write_up", END)
weather_subgraph = weather_builder.compile()


class TripState(TypedDict):
    question: str
    weather_report: str


parent_builder = StateGraph(TripState)
parent_builder.add_node("weather", weather_subgraph)    # the compiled graph IS the node
parent_builder.add_edge(START, "weather")
parent_builder.add_edge("weather", END)
parent_graph = parent_builder.compile()


def main() -> None:
    question = {"question": "Montreal this weekend?"}
    print("parent result:", parent_graph.invoke(question))
    # Run the subgraph by itself to see the private key the parent never got.
    print("subgraph alone:", weather_subgraph.invoke(question))


if __name__ == "__main__":
    main()
