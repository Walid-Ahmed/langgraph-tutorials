# Step 4a in 5-Workflows: dynamic fan-out with Send, with no model and no API key.
#
# This script splits a sentence into words and starts ONE worker per word.
# Each worker measures its own word and appends one line to a shared list,
# and a final node joins the lines. The number of workers is not written
# anywhere in the graph: it is decided at run time by the input.
#
#     START -> plan -> (one "measure" per word, in parallel) -> combine -> END
#
# Why this matters:
# - 03a_parallel_math.py fixed three branches with add_edge. Here the dispatch
#   function returns a list of Send objects, so a 3-word sentence gets 3
#   workers and a 6-word sentence gets 6, from the same compiled graph.
# - Each Send carries a private input, so a worker sees only its own word,
#   not the whole state.
# - All workers write the SAME field, so that field needs a reducer.
# - It does NOT call a model. 04_orchestrator_workers.py is the same shape
#   with an LLM planner and LLM workers.
#
# Before you run this:
# 1. pip install -r requirements.txt   (only langgraph is used; no API key)
#
# Run it from the repository root with:
#    python "5-Workflows/04a_send_minimal.py"
#
# Then run:
#    python "5-Workflows/04_orchestrator_workers.py"   (needs OPENAI_API_KEY)

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class OverallState(TypedDict):
    sentence: str
    words: list[str]                        # written once, by plan
    lengths: Annotated[list[str], add]      # every worker appends one line
    summary: str                            # written once, by combine


class WorkerState(TypedDict):
    word: str                               # a worker sees only its own word


def plan(state: OverallState) -> dict:
    return {"words": state["sentence"].split()}


def dispatch(state: OverallState) -> list[Send]:
    # One Send per planned item: "run measure once, with THIS input".
    return [Send("measure", {"word": word}) for word in state["words"]]


def measure(state: WorkerState) -> dict:
    # A one-item list: the add reducer concatenates the workers' lists.
    return {"lengths": [f"{state['word']}={len(state['word'])}"]}


def combine(state: OverallState) -> dict:
    return {"summary": ", ".join(state["lengths"])}


builder = StateGraph(OverallState)
builder.add_node("plan", plan)
builder.add_node("measure", measure)
builder.add_node("combine", combine)
builder.add_edge(START, "plan")
# The dispatch function replaces a router: it returns Sends, not a node name.
builder.add_conditional_edges("plan", dispatch, ["measure"])
builder.add_edge("measure", "combine")      # combine waits for every worker
builder.add_edge("combine", END)
graph = builder.compile()


if __name__ == "__main__":
    for sentence in ["graphs pass batons", "one worker per word decided at runtime"]:
        workers = 0
        summary = ""
        for update in graph.stream({"sentence": sentence}, stream_mode="updates"):
            workers += 1 if "measure" in update else 0
            summary = update.get("combine", {}).get("summary", summary)
        # Expected: 3 workers for the first sentence, 7 for the second.
        print(f"{sentence!r}\n  workers: {workers}\n  summary: {summary}")
