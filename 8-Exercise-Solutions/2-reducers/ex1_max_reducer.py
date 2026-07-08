# Exercise 1 — Max reducer
#
# high_score always keeps whichever value is higher.
# Starting at 42, a node returning 10 leaves it at 42.
# A node returning 100 updates it to 100.

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


def keep_max(current: int, new: int) -> int:
    return max(current, new)


class State(TypedDict):
    high_score: Annotated[int, keep_max]


def low_score_node(state: State) -> dict:
    print(f"Node returning 10 (current high_score={state['high_score']})")
    return {"high_score": 10}


def high_score_node(state: State) -> dict:
    print(f"Node returning 100 (current high_score={state['high_score']})")
    return {"high_score": 100}


def main() -> None:
    # Test 1: new value is lower — high_score should stay at 42
    graph = StateGraph(State)
    graph.add_node("update", low_score_node)
    graph.add_edge(START, "update")
    graph.add_edge("update", END)
    app = graph.compile()

    result = app.invoke({"high_score": 42})
    print(f"After low_score_node: high_score = {result['high_score']}")
    assert result["high_score"] == 42, "Expected 42"

    # Test 2: new value is higher — high_score should become 100
    graph2 = StateGraph(State)
    graph2.add_node("update", high_score_node)
    graph2.add_edge(START, "update")
    graph2.add_edge("update", END)
    app2 = graph2.compile()

    result2 = app2.invoke({"high_score": 42})
    print(f"After high_score_node: high_score = {result2['high_score']}")
    assert result2["high_score"] == 100, "Expected 100"


if __name__ == "__main__":
    main()
