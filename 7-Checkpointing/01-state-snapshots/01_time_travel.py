# Step 1 in 01-state-snapshots: replay and fork a run from an earlier checkpoint.
#
# This script runs a two-node graph (double a number, then add ten), reads the
# thread's checkpoint history, picks the checkpoint saved just before the
# second node, and continues from it twice: once unchanged (a replay) and once
# after editing the saved value (a fork).
#
# Why this matters:
# - Every checkpoint has its own checkpoint_id, so "continue from here" works
#   for any saved point, not only the latest one.
# - invoke(None, that_checkpoint's_config) re-runs only the nodes AFTER that
#   point. The "double" node does not run again.
# - update_state on an old checkpoint does not rewrite history. It saves a new
#   checkpoint beside the old one, and the original run is still there.
# - It does NOT call a model, so no API key is needed.
#
# Before you run this:
# 1. pip install -r requirements.txt   (only langgraph is used; no API key)
#
# Run it from the repository root with:
#    python "7-Checkpointing/01-state-snapshots/01_time_travel.py"

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    value: int


def double(state: State) -> dict:
    print("  running double")
    return {"value": state["value"] * 2}


def add_ten(state: State) -> dict:
    print("  running add_ten")
    return {"value": state["value"] + 10}


builder = StateGraph(State)
builder.add_node("double", double)
builder.add_node("add_ten", add_ten)
builder.add_edge(START, "double")
builder.add_edge("double", "add_ten")
builder.add_edge("add_ten", END)
graph = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "time-travel-demo"}}

    print("=== The original run ===")
    print("result:", graph.invoke({"value": 5}, config))

    # The history is newest first. Find the checkpoint whose next node is add_ten:
    # it was saved after double finished and before add_ten started.
    history = list(graph.get_state_history(config))
    before_add_ten = next(s for s in history if s.next == ("add_ten",))
    print("\ncheckpoint before add_ten holds:", before_add_ten.values)

    print("\n=== Replay from that checkpoint ===")
    print("result:", graph.invoke(None, before_add_ten.config))

    print("\n=== Fork: edit the saved value, then continue ===")
    forked_config = graph.update_state(before_add_ten.config, {"value": 100})
    print("result:", graph.invoke(None, forked_config))

    # The first run's checkpoint is unchanged: forking added to the history.
    print("\noriginal checkpoint still holds:", graph.get_state(before_add_ten.config).values)
    print("checkpoints on the thread now:", len(list(graph.get_state_history(config))))
