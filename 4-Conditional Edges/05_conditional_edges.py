# Demonstrates conditional routing: grade_node scores a fake "answer", then
# a router function (not a graph node) inspects the score and picks between
# two branches (pass_node vs retry_node) using add_conditional_edges.

import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph


# ---------------------------------------------------------
# State Schema
#
# This is the data that moves through the graph.
# Each node can read this state and return updates to it.
# ---------------------------------------------------------
class AgentState(TypedDict):
    answer: str
    score: int
    result: str


# ---------------------------------------------------------
# Grade Node
#
# This node checks the user's answer and gives it a score.
# After this node runs, the router decides where to go next.
# ---------------------------------------------------------
def grade_node(state: AgentState) -> dict:

    print("\nGrading answer:")
    print(state["answer"])

    # Simple fake grading logic:
    # If the answer mentions "RAG", give it a passing score.
    if "rag" in state["answer"].lower():
        score = 90
    else:
        score = 50

    return {
        "score": score
    }


# ---------------------------------------------------------
# Router Function
#
# This is NOT a graph node.
# It only reads the state and returns the name of the next node.
#
# If score >= 70  -> go to pass_node
# If score < 70   -> go to retry_node
# ---------------------------------------------------------
def route_after_grade(state: AgentState) -> str:

    if state["score"] >= 70:
        return "pass_node"

    return "retry_node"


# ---------------------------------------------------------
# Pass Node
#
# Runs when the score is high enough.
# ---------------------------------------------------------
def pass_node(state: AgentState) -> dict:
    return {
        "result": "Passed ✅"
    }


# ---------------------------------------------------------
# Retry Node
#
# Runs when the score is too low.
# ---------------------------------------------------------
def retry_node(state: AgentState) -> dict:
    return {
        "result": "Retry needed 🔁"
    }


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------
def main():

    # Create the graph using our state schema.
    graph = StateGraph(AgentState)

    # Add the three real graph nodes.
    graph.add_node("grade_node", grade_node)
    graph.add_node("pass_node", pass_node)
    graph.add_node("retry_node", retry_node)

    # Normal edge:
    # START always goes to grade_node.
    graph.add_edge(START, "grade_node")

    # Conditional edge:
    # After grade_node runs, route_after_grade decides the next node.
    graph.add_conditional_edges(
        "grade_node",          # source node
        route_after_grade,     # router function
        {
            "pass_node": "pass_node",
            "retry_node": "retry_node",
        }
    )

    # Both possible paths finish at END.
    graph.add_edge("pass_node", END)
    graph.add_edge("retry_node", END)

    # Compile the graph into an app we can run.
    app = graph.compile()

    # Print and save the graph visualization.
    plot_graph(app)

    # Try changing this answer to see the graph take a different route.
    initial_state = {
        "answer": "RAG means retrieval augmented generation.",
        "score": 0,
        "result": ""
    }

    final_state = app.invoke(initial_state)

    print("\nFinal State:")
    print(final_state)


if __name__ == "__main__":
    main()
