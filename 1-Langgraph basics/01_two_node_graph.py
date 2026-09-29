# Run from the repository root:
#   python "1-Langgraph basics/01_two_node_graph.py"
#
# Two runners, one baton: the smallest graph with a handoff between nodes.
# Converts Celsius to Fahrenheit (F = C * 9/5 + 32) in two steps:
#   scale  - multiplies celsius by 9/5 and writes `scaled`
#   shift  - reads `scaled`, adds 32 and writes `fahrenheit`
#
# What it demonstrates (book Chapter 2, Section 2.8):
#   - an edge between two nodes is a handoff: shift reads what scale wrote,
#     and neither function calls the other;
#   - each node returns only its own field (a partial update), so `celsius`
#     and `scaled` ride along untouched to the final state.
#
# Prerequisites: `pip install -r requirements.txt`. No API key needed.
#
# Expected output:
#   {'celsius': 25, 'scaled': 45.0, 'fahrenheit': 77.0}
#   {'celsius': 100, 'scaled': 180.0, 'fahrenheit': 212.0}

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class TempState(TypedDict):
    celsius: float       # the input
    scaled: float        # written by scale, read by shift
    fahrenheit: float    # the result


def scale(state: TempState) -> dict:
    # Return only the field this runner produces; LangGraph merges it in.
    return {"scaled": state["celsius"] * 9 / 5}


def shift(state: TempState) -> dict:
    # `scaled` is guaranteed to be there: the edge scale -> shift ran scale first.
    return {"fahrenheit": state["scaled"] + 32}


def build_graph():
    graph = StateGraph(TempState)
    graph.add_node("scale", scale)
    graph.add_node("shift", shift)
    graph.add_edge(START, "scale")
    graph.add_edge("scale", "shift")     # the handoff
    graph.add_edge("shift", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print(app.invoke({"celsius": 25}))
    print(app.invoke({"celsius": 100}))
