# Step 5a in 4-Conditional Edges: a router that returns labels, not node names.
#
# This script grades an answer with a keyword check, and its router returns
# the abstract labels "good" or "bad". The mapping passed to
# add_conditional_edges decides which node each label means. The graph is
# built twice with two different mappings and the SAME router.
#
# Why this matters:
# - In 05_conditional_edges.py the router returned node names, so the mapping
#   looked redundant. Here it does real work: the decision ("is it good?")
#   and the wiring ("where does good go?") live in different places.
# - Repointing "bad" from retry_node to review_node changes one dictionary
#   entry. The router is not touched.
# - It does NOT call a model, so no API key is needed.
#
# Before you run this:
# 1. pip install -r requirements.txt   (only langgraph is used; no API key)
#
# Run it from the repository root with:
#    python "4-Conditional Edges/05a_label_mapping.py"

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    answer: str
    score: int
    result: str


def grade_node(state: AgentState) -> dict:
    return {"score": 90 if "rag" in state["answer"].lower() else 50}


def pass_node(state: AgentState) -> dict:
    return {"result": "Passed"}


def retry_node(state: AgentState) -> dict:
    return {"result": "Retry needed"}


def review_node(state: AgentState) -> dict:
    return {"result": "Sent to a human reviewer"}


def route_after_grade(state: AgentState) -> str:
    # The router knows about quality. It does not know any node's name.
    return "good" if state["score"] >= 70 else "bad"


def build(mapping: dict[str, str]):
    graph = StateGraph(AgentState)
    graph.add_node("grade_node", grade_node)
    for node in (pass_node, retry_node, review_node):
        graph.add_node(node.__name__, node)
        graph.add_edge(node.__name__, END)
    graph.add_edge(START, "grade_node")
    # The mapping is the only place a label meets a node name.
    graph.add_conditional_edges("grade_node", route_after_grade, mapping)
    return graph.compile()


if __name__ == "__main__":
    weak_answer = {"answer": "I am not sure.", "score": 0, "result": ""}

    first = build({"good": "pass_node", "bad": "retry_node"})
    print("bad -> retry_node :", first.invoke(weak_answer)["result"])

    # Same router, one dictionary entry changed.
    second = build({"good": "pass_node", "bad": "review_node"})
    print("bad -> review_node:", second.invoke(weak_answer)["result"])
