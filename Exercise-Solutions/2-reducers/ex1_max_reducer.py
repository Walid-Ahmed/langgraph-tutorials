# Run from the repository root:
#   python "Exercise-Solutions/2-reducers/ex1_max_reducer.py"
#
# Exercise 3.2 — A "keep the maximum" reducer (Appendix B, Listing B.4)
#
# What it shows: a reducer is any function (current, new) -> combined.
# Two nodes run one after the other; round_two writes a LOWER score,
# but keep_max decides the merge, so the higher value survives.
# No API key needed.
#
# Expected output:
#   {'high_score': 72}

import sys
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph


def keep_max(current: int, new: int) -> int:
    return max(current, new)


class ScoreState(TypedDict):
    high_score: Annotated[int, keep_max]


def round_one(state): return {"high_score": 72}
def round_two(state): return {"high_score": 45}      # lower — must NOT win


def main() -> None:
    graph = StateGraph(ScoreState)
    graph.add_node("round_one", round_one)
    graph.add_node("round_two", round_two)
    graph.add_edge(START, "round_one")
    graph.add_edge("round_one", "round_two")
    graph.add_edge("round_two", END)
    app = graph.compile()
    plot_graph(app, print_mermaid=False)

    result = app.invoke({"high_score": 0})
    print(result)
    # Swap the order of the two nodes and the result is still 72:
    # the reducer, not the write order, decides.
    assert result["high_score"] == 72, "Expected 72"


if __name__ == "__main__":
    main()
