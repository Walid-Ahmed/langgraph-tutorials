# Exercise 3 — Structured-output router
#
# Step 1: classify_node uses a Pydantic schema to classify the input
#         into one of three categories: factual / creative / technical.
# Step 2: a conditional edge routes to a specialist LLM node.
# Each specialist node uses a tailored prompt for its category.

import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------
# Pydantic schema for classification output
# ---------------------------------------------------------
class Classification(BaseModel):
    category: Literal["factual", "creative", "technical"]


classifier_llm = llm.with_structured_output(Classification)


# ---------------------------------------------------------
# State
# ---------------------------------------------------------
class State(TypedDict):
    question: str
    category: str
    answer: str


# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------
def classify_node(state: State) -> dict:
    result: Classification = classifier_llm.invoke(
        f"Classify this question into one of: factual, creative, technical.\n\nQuestion: {state['question']}"
    )
    print(f"Classified as: {result.category}")
    return {"category": result.category}


def route_by_category(state: State) -> str:
    return state["category"]


def factual_node(state: State) -> dict:
    answer = llm.invoke(
        f"Answer this factual question concisely and accurately:\n\n{state['question']}"
    ).content
    return {"answer": answer}


def creative_node(state: State) -> dict:
    answer = llm.invoke(
        f"Answer this question imaginatively and creatively:\n\n{state['question']}"
    ).content
    return {"answer": answer}


def technical_node(state: State) -> dict:
    answer = llm.invoke(
        f"Answer this technical question with precise detail and, if helpful, code:\n\n{state['question']}"
    ).content
    return {"answer": answer}


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------
def main() -> None:
    graph = StateGraph(State)
    graph.add_node("classify", classify_node)
    graph.add_node("factual", factual_node)
    graph.add_node("creative", creative_node)
    graph.add_node("technical", technical_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_category,
        {"factual": "factual", "creative": "creative", "technical": "technical"},
    )
    graph.add_edge("factual", END)
    graph.add_edge("creative", END)
    graph.add_edge("technical", END)

    app = graph.compile()
    plot_graph(app)

    questions = [
        "What year did the Berlin Wall fall?",
        "Write a metaphor for loneliness.",
        "How does Python's GIL affect multithreaded performance?",
    ]

    for q in questions:
        result = app.invoke({"question": q, "category": "", "answer": ""})
        print(f"\nQ: {q}")
        print(f"Category: {result['category']}")
        print(f"A: {result['answer'][:200]}...")


if __name__ == "__main__":
    main()
