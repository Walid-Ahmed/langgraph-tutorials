# Exercise 3 — Input validation with a conditional edge
#
# validate_node runs first. If input is empty it sets an error message
# and routes straight to END. Otherwise it continues to process_node.

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


def validate_node(state: SimpleState) -> dict:
    if not state["input"].strip():
        return {"output": "error: empty input"}
    return {}


def route_after_validate(state: SimpleState) -> str:
    if state["output"].startswith("error"):
        return END
    return "process"


def process(state: SimpleState) -> dict:
    return {
        "output": state["input"].upper(),
        "step": state["step"] + 1,
    }


def main() -> None:
    graph = StateGraph(SimpleState)
    graph.add_node("validate", validate_node)
    graph.add_node("process", process)

    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"process": "process", END: END})
    graph.add_edge("process", END)

    app = graph.compile()
    plot_graph(app)

    print("Valid input:")
    print(app.invoke({"input": "hello", "output": "", "step": 0}))
    # Expected: output = 'HELLO'

    print("\nEmpty input:")
    print(app.invoke({"input": "", "output": "", "step": 0}))
    # Expected: output = 'error: empty input'


if __name__ == "__main__":
    main()
