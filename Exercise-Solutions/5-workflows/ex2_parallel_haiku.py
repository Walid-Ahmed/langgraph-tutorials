# Exercise 2 — Add a haiku branch to the parallelization example
#
# Extends 03_parallelization_creative.py with a fourth parallel branch.
# joke / story / poem / haiku all run concurrently from the same topic.

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
    joke: str
    story: str
    poem: str
    haiku: str
    combined_output: str


def generate_joke(state: State) -> dict:
    return {"joke": llm.invoke(f"Write a short joke about {state['topic']}").content}


def generate_story(state: State) -> dict:
    return {"story": llm.invoke(f"Write a short story about {state['topic']}").content}


def generate_poem(state: State) -> dict:
    return {"poem": llm.invoke(f"Write a short poem about {state['topic']}").content}


def generate_haiku(state: State) -> dict:
    return {"haiku": llm.invoke(f"Write a haiku about {state['topic']}").content}


def aggregator(state: State) -> dict:
    combined = (
        f"Everything about {state['topic']}:\n\n"
        f"STORY:\n{state['story']}\n\n"
        f"JOKE:\n{state['joke']}\n\n"
        f"POEM:\n{state['poem']}\n\n"
        f"HAIKU:\n{state['haiku']}"
    )
    return {"combined_output": combined}


def main() -> None:
    graph = StateGraph(State)
    graph.add_node("generate_joke", generate_joke)
    graph.add_node("generate_story", generate_story)
    graph.add_node("generate_poem", generate_poem)
    graph.add_node("generate_haiku", generate_haiku)
    graph.add_node("aggregator", aggregator)

    for node in ("generate_joke", "generate_story", "generate_poem", "generate_haiku"):
        graph.add_edge(START, node)
        graph.add_edge(node, "aggregator")

    graph.add_edge("aggregator", END)
    app = graph.compile()
    plot_graph(app)

    result = app.invoke({
        "topic": "cats",
        "joke": "", "story": "", "poem": "", "haiku": "", "combined_output": "",
    })
    print(result["combined_output"])


if __name__ == "__main__":
    main()
