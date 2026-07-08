# Exercise 2 — Track visited nodes
#
# Add a "visited" field (list of strings) to the state.
# Each node appends its own name when it runs.
# Without a reducer, returning a new list replaces the old one —
# so we attach an "add" reducer to merge lists instead.

import sys
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph


class SimpleState(TypedDict):
    input: str
    output: str
    step: int
    # The "add" reducer concatenates lists, so each node can safely
    # return ["node_name"] without wiping out previous entries.
    visited: Annotated[list, add]


def process(state: SimpleState) -> dict:
    return {
        "output": state["input"].upper(),
        "step": state["step"] + 1,
        "visited": ["process"],
    }


def reverse_node(state: SimpleState) -> dict:
    return {
        "output": state["output"][::-1],
        "step": state["step"] + 1,
        "visited": ["reverse"],
    }


def main() -> None:
    graph = StateGraph(SimpleState)
    graph.add_node("process", process)
    graph.add_node("reverse", reverse_node)

    graph.add_edge(START, "process")
    graph.add_edge("process", "reverse")
    graph.add_edge("reverse", END)

    app = graph.compile()
    plot_graph(app)

    result = app.invoke({"input": "hello", "output": "", "step": 0, "visited": []})
    print(result)
    # Expected: visited = ['process', 'reverse']


if __name__ == "__main__":
    main()
