# Exercise 2 — Deduplicate list reducer
#
# animals appends new items but skips ones already in the list.
# ["lion", "tiger"] + ["tiger", "cat"] => ["lion", "tiger", "cat"]

import sys
from pathlib import Path
from typing import Annotated, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph


def append_unique(current: List[str], new: List[str]) -> List[str]:
    seen = set(current)
    return current + [item for item in new if item not in seen]


class State(TypedDict):
    animals: Annotated[List[str], append_unique]


def update_node(state: State) -> dict:
    print(f"Current animals: {state['animals']}")
    return {"animals": ["tiger", "cat"]}


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("update", update_node)
    graph.add_edge(START, "update")
    graph.add_edge("update", END)
    app = graph.compile()
    plot_graph(app)

    result = app.invoke({"animals": ["lion", "tiger"]})
    print(f"Final animals: {result['animals']}")
    assert result["animals"] == ["lion", "tiger", "cat"], f"Unexpected: {result['animals']}"


if __name__ == "__main__":
    main()
