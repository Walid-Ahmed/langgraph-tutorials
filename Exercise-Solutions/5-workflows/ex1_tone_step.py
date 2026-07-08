# Exercise 1 — Extend the prompt chain with a casual tone step
#
# Adds a fifth node after improve_content that rewrites the draft in
# a casual, conversational tone and stores it in casual_version.
# The final state contains both improved_content and casual_version.

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


class ContentState(TypedDict):
    topic: str
    requirements: str
    draft: str
    fact_check_results: str
    improved_content: str
    casual_version: str


def generate_draft(state: ContentState) -> dict:
    prompt = f"Write a 150-word blog post about: {state['topic']}. Requirements: {state['requirements']}"
    draft = llm.invoke(prompt).content
    print("=== STEP 1: Draft ===")
    print(draft[:100] + "...\n")
    return {"draft": draft}


def fact_check(state: ContentState) -> dict:
    prompt = f"Briefly identify any factual issues in this draft:\n\n{state['draft']}"
    result = llm.invoke(prompt).content
    print("=== STEP 2: Fact check ===")
    print(result[:100] + "...\n")
    return {"fact_check_results": result}


def improve_content(state: ContentState) -> dict:
    prompt = (
        f"Revise this draft based on the feedback.\n\n"
        f"Draft:\n{state['draft']}\n\n"
        f"Feedback:\n{state['fact_check_results']}\n\n"
        "Keep it around 150 words."
    )
    improved = llm.invoke(prompt).content
    print("=== STEP 3: Improved ===")
    print(improved[:100] + "...\n")
    return {"improved_content": improved}


def casual_tone(state: ContentState) -> dict:
    prompt = (
        "Rewrite the following blog post in a casual, conversational tone "
        "as if you're texting a friend. Keep it around 150 words.\n\n"
        f"{state['improved_content']}"
    )
    casual = llm.invoke(prompt).content
    print("=== STEP 4: Casual version ===")
    print(casual[:100] + "...\n")
    return {"casual_version": casual}


def main() -> None:
    graph = StateGraph(ContentState)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("fact_check", fact_check)
    graph.add_node("improve_content", improve_content)
    graph.add_node("casual_tone", casual_tone)

    graph.add_edge(START, "generate_draft")
    graph.add_edge("generate_draft", "fact_check")
    graph.add_edge("fact_check", "improve_content")
    graph.add_edge("improve_content", "casual_tone")
    graph.add_edge("casual_tone", END)

    app = graph.compile()
    plot_graph(app)

    result = app.invoke({
        "topic": "The benefits of morning exercise",
        "requirements": "Target audience: busy professionals",
        "draft": "",
        "fact_check_results": "",
        "improved_content": "",
        "casual_version": "",
    })

    print("\n=== FINAL: Improved content ===")
    print(result["improved_content"])
    print("\n=== FINAL: Casual version ===")
    print(result["casual_version"])


if __name__ == "__main__":
    main()
