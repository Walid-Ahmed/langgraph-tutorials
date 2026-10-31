# Run from the repository root:
#   python "5-Workflows/03a_parallel_math.py"
#
# What it does:
#   Takes two numbers, x and y, and runs three nodes IN PARALLEL on them —
#   add, subtract, multiply — then a fourth node, combine, waits for all three
#   and writes a one-line summary. Then it repeats the fan-out with all three
#   nodes writing ONE shared field, first without a reducer, then with one.
#
# What it demonstrates (book Chapter 5, Section 5.6):
#   - static fan-out: several edges leave START, so their nodes run in the same step
#   - fan-in: an edge from each branch into combine; combine waits for all of them
#   - distinct fields need no reducer (each branch owns its own key)
#   - a SHARED field written in parallel raises InvalidUpdateError without a
#     reducer, and combines cleanly with Annotated[list, operator.add]
#
# Prerequisites: `pip install -r requirements.txt`. No API key, no model call.
#
# Expected output:
#   final state: {'x': 6, 'y': 3, 'total': 9, 'difference': 3, 'product': 18,
#                 'summary': 'sum=9, difference=3, product=18'}
#   step by step: add, multiply and subtract in one step (any order), then combine
#   shared field, no reducer   -> InvalidUpdateError ...
#   shared field, with reducer -> results: ['6 + 3 = 9', '6 - 3 = 3', '6 x 3 = 18'] (any order)

from operator import add
from typing import Annotated

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ------------------------------------------------ 1. three branches, three fields
class MathState(TypedDict):
    x: float
    y: float
    total: float        # written only by add
    difference: float   # written only by subtract
    product: float      # written only by multiply
    summary: str        # written only by combine


def add_node(state: MathState) -> dict:
    return {"total": state["x"] + state["y"]}


def subtract_node(state: MathState) -> dict:
    return {"difference": state["x"] - state["y"]}


def multiply_node(state: MathState) -> dict:
    return {"product": state["x"] * state["y"]}


def combine_node(state: MathState) -> dict:
    # Safe to read all three: the fan-in edges make combine wait for every branch.
    return {"summary": f"sum={state['total']}, difference={state['difference']}, "
                       f"product={state['product']}"}


def build_math_graph():
    graph = StateGraph(MathState)
    graph.add_node("add", add_node)
    graph.add_node("subtract", subtract_node)
    graph.add_node("multiply", multiply_node)
    graph.add_node("combine", combine_node)
    for branch in ("add", "subtract", "multiply"):
        graph.add_edge(START, branch)       # fan-out: all three start together
        graph.add_edge(branch, "combine")   # fan-in: combine waits for all three
    graph.add_edge("combine", END)
    return graph.compile()


# ------------------------------------ 2. the same fan-out, one shared field
class SharedNoReducer(TypedDict):
    x: float
    y: float
    results: list[str]                     # no reducer: one value per step, please


class SharedWithReducer(TypedDict):
    x: float
    y: float
    results: Annotated[list[str], add]     # reducer: concatenate every write


def add_line(state) -> dict:
    return {"results": [f"{state['x']} + {state['y']} = {state['x'] + state['y']}"]}


def subtract_line(state) -> dict:
    return {"results": [f"{state['x']} - {state['y']} = {state['x'] - state['y']}"]}


def multiply_line(state) -> dict:
    return {"results": [f"{state['x']} x {state['y']} = {state['x'] * state['y']}"]}


def build_shared_graph(schema):
    graph = StateGraph(schema)
    for name, fn in (("add", add_line), ("subtract", subtract_line), ("multiply", multiply_line)):
        graph.add_node(name, fn)
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    return graph.compile()


def main() -> None:
    inputs = {"x": 6, "y": 3}

    app = build_math_graph()
    print("final state:", app.invoke(inputs))

    print("\nstep by step (stream_mode='updates'):")
    for update in app.stream(inputs, stream_mode="updates"):
        print(" ", update)

    print("\nshared field, no reducer:")
    try:
        build_shared_graph(SharedNoReducer).invoke({**inputs, "results": []})
    except InvalidUpdateError as error:
        print("  InvalidUpdateError:", str(error).splitlines()[0])

    print("\nshared field, with reducer:")
    result = build_shared_graph(SharedWithReducer).invoke({**inputs, "results": []})
    print("  results:", result["results"])


if __name__ == "__main__":
    main()
