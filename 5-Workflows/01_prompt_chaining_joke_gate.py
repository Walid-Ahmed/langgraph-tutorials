# Prompt chaining with a quality gate: after generate_joke, a router
# function (check_punchline) inspects the output and either ends the chain
# early ("Pass") or routes through improve_joke -> polish_joke ("Fail").

import sys
from pathlib import Path
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------
# 1. State Definition
#
# This prompt chain starts with a topic, generates a joke,
# optionally improves it, then optionally polishes it.
#
# Node returns are partial state updates. For example,
# returning {"joke": msg.content} updates only joke and
# keeps topic unchanged. No reducer is used here.
# ---------------------------------------------------------
class JokeState(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


# ---------------------------------------------------------
# 2. Nodes
# ---------------------------------------------------------
def generate_joke(state: JokeState) -> dict:
    """First LLM call: generate an initial joke."""

    msg = llm.invoke(f"Write a short joke about {state['topic']}.")
    return {"joke": msg.content}


def improve_joke(state: JokeState) -> dict:
    """Second LLM call: improve the joke with wordplay."""

    msg = llm.invoke(
        f"Make this joke funnier by adding wordplay:\n\n{state['joke']}"
    )
    return {"improved_joke": msg.content}


def polish_joke(state: JokeState) -> dict:
    """Third LLM call: add a final surprising twist."""

    msg = llm.invoke(
        f"Add a surprising twist to this joke:\n\n{state['improved_joke']}"
    )
    return {"final_joke": msg.content}


# ---------------------------------------------------------
# 3. Conditional Gate
#
# This is not a node that updates state. It is a router.
# It reads the joke and returns the next path label.
# ---------------------------------------------------------
def check_punchline(state: JokeState) -> str:
    """Route based on whether the joke seems to have a punchline."""

    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"

    return "Fail"


# ---------------------------------------------------------
# 4. Build The Graph
#
# Flow:
# START -> generate_joke -> check_punchline
#
# If Pass: END
# If Fail: improve_joke -> polish_joke -> END
# ---------------------------------------------------------
graph_builder = StateGraph(JokeState)

graph_builder.add_node("generate_joke", generate_joke)
graph_builder.add_node("improve_joke", improve_joke)
graph_builder.add_node("polish_joke", polish_joke)

graph_builder.add_edge(START, "generate_joke")
graph_builder.add_conditional_edges(
    "generate_joke",
    check_punchline,
    {
        "Fail": "improve_joke",
        "Pass": END,
    },
)
graph_builder.add_edge("improve_joke", "polish_joke")
graph_builder.add_edge("polish_joke", END)

graph = graph_builder.compile()


# ---------------------------------------------------------
# 5. Run It
# ---------------------------------------------------------
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "01_prompt_chaining_joke_gate_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(graph, graph_image_path)

    state = graph.invoke({"topic": "cats"})

    print("Initial joke:")
    print(state["joke"])
    print("\n--- --- ---\n")

    if "improved_joke" in state:
        print("Improved joke:")
        print(state["improved_joke"])
        print("\n--- --- ---\n")

        print("Final joke:")
        print(state["final_joke"])
    else:
        print("Final joke:")
        print(state["joke"])


if __name__ == "__main__":
    main()
