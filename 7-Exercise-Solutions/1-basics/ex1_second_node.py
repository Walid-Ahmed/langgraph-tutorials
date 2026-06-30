# Exercise 1 — Add a second node
#
# Extend the simple graph with a reverse_node that runs after process.
# Path: START -> process -> reverse -> END
# Input "hello" should produce output "OLLEH".

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph


class SimpleState(TypedDict):
    input: str
    output: str
    step: int


def process(state: SimpleState) -> dict:
    return {
        "output": state["input"].upper(),
        "step": state["step"] + 1,
    }


def reverse_node(state: SimpleState) -> dict:
    return {
        "output": state["output"][::-1],
        "step": state["step"] + 1,
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

    result = app.invoke({"input": "hello", "output": "", "step": 0})
    print(result)
    # Expected: {'input': 'hello', 'output': 'OLLEH', 'step': 2}


if __name__ == "__main__":
    main()
