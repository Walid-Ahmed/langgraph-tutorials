import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()


class SimpleState(TypedDict):
    input: str
    output: str
    step: int

def process(state: SimpleState) -> dict:
    output = state["input"].upper()
    step = state["step"] + 1
    return {
        "output": output,
        "step": step,
    }


def demo_simple_graph() -> None:
    # Create graph
    graph = StateGraph(SimpleState)
    # Add nodes
    graph.add_node("process", process)
    # Add edges
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    # Execute graph / compile
    app = graph.compile()

    # Print and save the graph visualization.
    plot_graph(app)

    # Run app
    initial_state = {
        "input": "hello",
        "output": "",
        "step": 0,
    }
    result = app.invoke(initial_state)

    print("Simple graph result:", result)

    print(
        f" Input: {result['input']}, "
        f"Output: {result['output']}, "
        f"Step: {result['step']}"
    )


if __name__ == "__main__":
    demo_simple_graph()
