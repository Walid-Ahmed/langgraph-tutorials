# Run from the repository root:
#   python "Exercise-Solutions/5-workflows/ex1_tone_step.py"
#
# Exercise 5.1 — Extend a chain (Appendix B, Listing B.11)
#
# What it shows: adding a stage to a prompt chain is one new node and one
# new state field. make_casual runs after improve_content and writes only
# casual_version, so no reducer is needed: no two nodes write the same key.
# Rule of thumb for chains: one stage, one new field.
# Needs OPENAI_API_KEY in the repository-root .env file.
#
# Expected output: the improved post, then a casual rewrite of it
# (text varies by model run).

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class State(TypedDict):
    topic: str
    content: str
    improved_content: str
    casual_version: str                  # new field, written by exactly one node


def generate_content(state: State) -> dict:
    msg = llm.invoke(f"Write a 120-word blog post about: {state['topic']}")
    return {"content": msg.content}


def improve_content(state: State) -> dict:
    msg = llm.invoke("Improve this post: fix any factual issues and tighten the wording. "
                     f"Keep it around 120 words.\n\n{state['content']}")
    return {"improved_content": msg.content}


def make_casual(state: State) -> dict:
    msg = llm.invoke("Rewrite this in a relaxed, casual tone, keeping every fact:\n\n"
                     f"{state['improved_content']}")
    return {"casual_version": msg.content}


def main() -> None:
    builder = StateGraph(State)
    builder.add_node("generate_content", generate_content)
    builder.add_node("improve_content", improve_content)
    builder.add_node("make_casual", make_casual)

    builder.add_edge(START, "generate_content")
    builder.add_edge("generate_content", "improve_content")
    builder.add_edge("improve_content", "make_casual")    # was: improve_content -> END
    builder.add_edge("make_casual", END)

    app = builder.compile()
    plot_graph(app, print_mermaid=False)

    result = app.invoke({"topic": "The benefits of morning exercise"})

    print("=== Improved content ===")
    print(result["improved_content"])
    print("\n=== Casual version ===")
    print(result["casual_version"])


if __name__ == "__main__":
    main()
