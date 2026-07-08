# Parallelization variant: START fans out into three independent creative
# writing nodes (joke, story, poem) for the same topic, then an aggregator
# node joins all three once they finish.

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()


# ---------------------------------------------------------
# 1. State Definition
#
# One topic enters the graph. Three independent nodes create
# different outputs from that same topic: a joke, story, poem.
#
# No reducer is needed because every parallel branch writes to
# a different state field.
# ---------------------------------------------------------
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------
# 2. Parallel LLM Nodes
#
# These nodes can run independently because none of them needs
# the output from another node.
# ---------------------------------------------------------
def generate_joke(state: State) -> dict:
    """Generate a joke about the topic."""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def generate_story(state: State) -> dict:
    """Generate a short story about the topic."""
    msg = llm.invoke(f"Write a short story about {state['topic']}")
    return {"story": msg.content}


def generate_poem(state: State) -> dict:
    """Generate a poem about the topic."""
    msg = llm.invoke(f"Write a short poem about {state['topic']}")
    return {"poem": msg.content}


# ---------------------------------------------------------
# 3. Aggregator Node
#
# The aggregator joins the parallel results into one response.
# ---------------------------------------------------------
def aggregator(state: State) -> dict:
    """Combine the joke, story, and poem into a single output."""
    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


# ---------------------------------------------------------
# 4. Build The Graph
#
# START fans out to three LLM nodes.
# The three outputs fan in to the aggregator.
# ---------------------------------------------------------
parallel_builder = StateGraph(State)

parallel_builder.add_node("generate_joke", generate_joke)
parallel_builder.add_node("generate_story", generate_story)
parallel_builder.add_node("generate_poem", generate_poem)
parallel_builder.add_node("aggregator", aggregator)

parallel_builder.add_edge(START, "generate_joke")
parallel_builder.add_edge(START, "generate_story")
parallel_builder.add_edge(START, "generate_poem")

parallel_builder.add_edge("generate_joke", "aggregator")
parallel_builder.add_edge("generate_story", "aggregator")
parallel_builder.add_edge("generate_poem", "aggregator")

parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()


# ---------------------------------------------------------
# 5. Run It
# ---------------------------------------------------------
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "03_parallelization_creative_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(parallel_workflow, graph_image_path)

    result = parallel_workflow.invoke(
        {
            "topic": "cats",
            "joke": "",
            "story": "",
            "poem": "",
            "combined_output": "",
        }
    )

    print(result["combined_output"])


if __name__ == "__main__":
    main()
