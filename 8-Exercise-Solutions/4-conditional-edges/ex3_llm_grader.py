# Exercise 3 — LLM-based grader
#
# Replace the keyword-matching score with a real LLM call.
# The LLM is asked to score the answer 0-100; we parse the integer.
# Routing logic (pass >= 70, else retry) stays the same.

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
    prompt = (
        "Score the following answer to the question 'What is RAG?' from 0 to 100. "
        "Respond with only the integer number, nothing else.\n\n"
        f"Answer: {state['answer']}"
    )
    response = llm.invoke(prompt).content.strip()
    try:
        score = int(response)
    except ValueError:
        score = 0
    print(f"LLM score: {score}")
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
