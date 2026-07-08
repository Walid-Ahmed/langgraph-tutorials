# Exercise 3 — Accumulate count across two nodes
#
# Both node_a and node_b return {"count": 5}.
# The custom_increment reducer adds values, so the final count is 10.
# This shows reducers apply across every node update, not just the last one.

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


def custom_increment(current: int, new: int) -> int:
    return current + new


class State(TypedDict):
    count: Annotated[int, custom_increment]


def node_a(state: State) -> dict:
    print(f"node_a: count before = {state['count']}, returning 5")
    return {"count": 5}


def node_b(state: State) -> dict:
    print(f"node_b: count before = {state['count']}, returning 5")
    return {"count": 5}


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)

    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)

    app = graph.compile()

    result = app.invoke({"count": 0})
    print(f"Final count: {result['count']}")
    assert result["count"] == 10, f"Expected 10, got {result['count']}"


if __name__ == "__main__":
    main()
