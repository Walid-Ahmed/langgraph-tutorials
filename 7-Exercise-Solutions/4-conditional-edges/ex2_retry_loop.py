# Exercise 2 — Retry loop with attempt limit
#
# retry_node loops back to grade_node instead of ending.
# A second conditional edge after grade_node stops the loop
# once attempts >= 3, regardless of the score.
#
# This is the foundation of an evaluator-optimizer pattern.

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


def grade_node(state: AgentState) -> dict:
    attempts = state["attempts"] + 1
    score = 90 if "rag" in state["answer"].lower() else 50
    print(f"Attempt {attempts}: score={score}")
    return {"score": score, "attempts": attempts}


def route_after_grade(state: AgentState) -> str:
    if state["attempts"] >= 3:
        return "end_node"
    if state["score"] >= 70:
        return "pass_node"
    return "retry_node"


def pass_node(state: AgentState) -> dict:
    return {"result": f"Passed ✅ (attempt {state['attempts']})"}


def retry_node(state: AgentState) -> dict:
    print("Retrying...")
    return {"result": "Retry needed 🔁"}


def end_node(state: AgentState) -> dict:
    return {"result": f"Max attempts reached after {state['attempts']} tries"}


def main() -> None:
    graph = StateGraph(AgentState)
    graph.add_node("grade_node", grade_node)
    graph.add_node("pass_node", pass_node)
    graph.add_node("retry_node", retry_node)
    graph.add_node("end_node", end_node)

    graph.add_edge(START, "grade_node")
    graph.add_conditional_edges(
        "grade_node",
        route_after_grade,
        {"pass_node": "pass_node", "retry_node": "retry_node", "end_node": "end_node"},
    )
    # retry loops back
    graph.add_edge("retry_node", "grade_node")
    graph.add_edge("pass_node", END)
    graph.add_edge("end_node", END)

    app = graph.compile()
    plot_graph(app)

    print("\n--- Failing answer (will hit attempt limit) ---")
    result = app.invoke({"answer": "I am not sure.", "score": 0, "result": "", "attempts": 0})
    print(f"Result: {result['result']}")

    print("\n--- Passing answer ---")
    result = app.invoke({"answer": "RAG means retrieval augmented generation.", "score": 0, "result": "", "attempts": 0})
    print(f"Result: {result['result']}")


if __name__ == "__main__":
    main()
