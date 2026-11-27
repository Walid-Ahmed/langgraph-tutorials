# Run from the repository root:
#   python "Exercise-Solutions/4-conditional-edges/ex1_three_branches.py"
#
# Exercise 4.1 — A third branch (Appendix B, Listing B.8)
#
# What it shows: adding an outcome takes three edits in three places:
# the evidence (grade_node), the decision (route_after_grade), and the
# topology (the path map plus an exit edge for the new node).
#   score >= 70 -> pass_node    ("Passed ✅")
#   60-69       -> review_node  ("Needs human review 👀")
#   < 60        -> retry_node   ("Retry needed 🔁")
# No API key needed.
#
# Expected output:
#   'RAG combines retrieval and generation' -> 90 -> Passed ✅
#   'It uses retrieval'                     -> 65 -> Needs human review 👀
#   'It is a database'                      -> 50 -> Retry needed 🔁

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


def grade_node(state: AgentState) -> dict:
    text = state["answer"].lower()
    if "rag" in text:
        score = 90
    elif "retrieval" in text and "generation" not in text:
        score = 65                                   # borderline
    else:
        score = 50
    return {"score": score}


def pass_node(state: AgentState) -> dict:
    return {"result": "Passed ✅"}


def review_node(state: AgentState) -> dict:
    return {"result": "Needs human review 👀"}


def retry_node(state: AgentState) -> dict:
    return {"result": "Retry needed 🔁"}


def route_after_grade(state: AgentState) -> str:
    if state["score"] >= 70:
        return "pass_node"
    elif state["score"] >= 60:
        return "review_node"
    return "retry_node"


def main() -> None:
    graph = StateGraph(AgentState)
    graph.add_node("grade_node", grade_node)
    graph.add_node("pass_node", pass_node)
    graph.add_node("review_node", review_node)
    graph.add_node("retry_node", retry_node)

    graph.add_edge(START, "grade_node")
    graph.add_conditional_edges("grade_node", route_after_grade,
        {"pass_node": "pass_node", "review_node": "review_node", "retry_node": "retry_node"})
    graph.add_edge("pass_node", END)
    graph.add_edge("review_node", END)                  # the new branch needs an exit too
    graph.add_edge("retry_node", END)

    app = graph.compile()
    plot_graph(app, print_mermaid=False)

    answers = ["RAG combines retrieval and generation", "It uses retrieval", "It is a database"]
    for answer in answers:
        result = app.invoke({"answer": answer, "score": 0, "result": ""})
        print(f"{repr(answer):<39} -> {result['score']} -> {result['result']}")


if __name__ == "__main__":
    main()
