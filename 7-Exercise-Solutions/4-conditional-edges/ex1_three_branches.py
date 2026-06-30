# Exercise 1 — Three branches: pass / needs_review / retry
#
# score >= 70  -> pass_node     ("Passed ✅")
# 60-69        -> review_node   ("Needs human review 👀")
# < 60         -> retry_node    ("Retry needed 🔁")
#
# Answers containing "retrieval" but not "generation" score 65 (review branch).

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
    answer = state["answer"].lower()
    if "rag" in answer or ("retrieval" in answer and "generation" in answer):
        score = 90
    elif "retrieval" in answer:
        score = 65
    else:
        score = 50
    print(f"Score: {score}")
    return {"score": score}


def route_after_grade(state: AgentState) -> str:
    if state["score"] >= 70:
        return "pass_node"
    if state["score"] >= 60:
        return "review_node"
    return "retry_node"


def pass_node(state: AgentState) -> dict:
    return {"result": "Passed ✅"}


def review_node(state: AgentState) -> dict:
    return {"result": "Needs human review 👀"}


def retry_node(state: AgentState) -> dict:
    return {"result": "Retry needed 🔁"}


def main() -> None:
    graph = StateGraph(AgentState)
    graph.add_node("grade_node", grade_node)
    graph.add_node("pass_node", pass_node)
    graph.add_node("review_node", review_node)
    graph.add_node("retry_node", retry_node)

    graph.add_edge(START, "grade_node")
    graph.add_conditional_edges(
        "grade_node",
        route_after_grade,
        {"pass_node": "pass_node", "review_node": "review_node", "retry_node": "retry_node"},
    )
    graph.add_edge("pass_node", END)
    graph.add_edge("review_node", END)
    graph.add_edge("retry_node", END)

    app = graph.compile()
    plot_graph(app)

    for answer in [
        "RAG means retrieval augmented generation.",   # pass
        "It involves retrieval from a knowledge base.", # review
        "I am not sure.",                              # retry
    ]:
        result = app.invoke({"answer": answer, "score": 0, "result": ""})
        print(f"Answer: '{answer[:40]}...' -> {result['result']}")


if __name__ == "__main__":
    main()
