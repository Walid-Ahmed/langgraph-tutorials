# Run from the repository root:
#   python "Exercise-Solutions/4-conditional-edges/ex2_retry_loop.py"
#
# Exercise 4.2 — Turn it into a retry loop (Appendix B, Listing B.9)
#
# What it shows: a loop needs a counter in state and a router with two
# exits. retry_node loops back to grade_node; once attempts reaches 3 the
# router returns "give_up", which maps straight to END, so the run stops
# by design instead of crashing into the recursion limit.
# No API key needed.
#
# Expected output (an answer that never passes, then one that does):
#   {'answer': 'It is a database', 'score': 50, 'result': 'Retry needed 🔁 (attempt 2)', 'attempts': 3}
#   {'answer': 'RAG combines retrieval and generation', 'score': 90, 'result': 'Passed ✅ (attempt 1)', 'attempts': 1}

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph


class AgentState(TypedDict):
    answer: str
    score: int
    result: str
    attempts: int


def grade_node(state):
    score = 90 if "rag" in state["answer"].lower() else 50
    return {"score": score, "attempts": state["attempts"] + 1}


def pass_node(state):
    return {"result": f"Passed ✅ (attempt {state['attempts']})"}


def retry_node(state):
    return {"result": f"Retry needed 🔁 (attempt {state['attempts']})"}


def route_after_grade(state):
    if state["score"] >= 70:
        return "pass_node"
    if state["attempts"] >= 3:
        return "give_up"                       # the guarantee
    return "retry_node"


def main() -> None:
    graph = StateGraph(AgentState)
    graph.add_node("grade_node", grade_node)
    graph.add_node("pass_node", pass_node)
    graph.add_node("retry_node", retry_node)

    graph.add_edge(START, "grade_node")
    graph.add_conditional_edges("grade_node", route_after_grade,
        {"pass_node": "pass_node", "retry_node": "retry_node", "give_up": END})
    graph.add_edge("retry_node", "grade_node")     # the back edge
    graph.add_edge("pass_node", END)

    app = graph.compile()
    plot_graph(app, print_mermaid=False)

    for answer in ["It is a database", "RAG combines retrieval and generation"]:
        print(app.invoke({"answer": answer, "score": 0, "result": "", "attempts": 0}))


if __name__ == "__main__":
    main()
