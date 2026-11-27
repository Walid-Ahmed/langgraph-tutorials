# Run from the repository root:
#   python "Exercise-Solutions/4-conditional-edges/ex3_llm_grader.py"
#
# Exercise 4.3 — A real LLM grader (Appendix B, Listing B.10)
#
# What it shows: the model writes the evidence (a 0-100 score); the router
# still just reads it, so the routing logic (pass >= 70, else retry) is
# unchanged. An unparseable reply falls back to 0: fail safe.
# Needs OPENAI_API_KEY in the repository-root .env file.
#
# Expected output (scores vary by model run):
#   -> Passed ✅ (score 100)
#   -> Retry needed 🔁 (score 20)
#   -> Retry needed 🔁 (score 0)

import sys
from pathlib import Path
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class AgentState(TypedDict):
    answer: str
    score: int
    result: str


def grade_node(state: AgentState) -> dict:
    reply = llm.invoke(
        "Score this answer to 'What is RAG?' from 0-100. "
        f"Respond with only the number.\n\nAnswer: {state['answer']}"
    ).content
    try:
        score = int(reply.strip())
    except ValueError:
        score = 0                                   # unparseable => fail safe
    return {"score": score}


def route_after_grade(state: AgentState) -> str:
    return "pass_node" if state["score"] >= 70 else "retry_node"


def pass_node(state: AgentState) -> dict:
    return {"result": f"Passed ✅ (score {state['score']})"}


def retry_node(state: AgentState) -> dict:
    return {"result": f"Retry needed 🔁 (score {state['score']})"}


def main() -> None:
    graph = StateGraph(AgentState)
    graph.add_node("grade_node", grade_node)
    graph.add_node("pass_node", pass_node)
    graph.add_node("retry_node", retry_node)

    graph.add_edge(START, "grade_node")
    graph.add_conditional_edges(
        "grade_node",
        route_after_grade,
        {"pass_node": "pass_node", "retry_node": "retry_node"},
    )
    graph.add_edge("pass_node", END)
    graph.add_edge("retry_node", END)

    app = graph.compile()
    plot_graph(app)

    for answer in [
        "RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a language model.",
        "I think RAG is something to do with computers.",
        "I don't know.",
    ]:
        result = app.invoke({"answer": answer, "score": 0, "result": ""})
        print(f"-> {result['result']}")


if __name__ == "__main__":
    main()
