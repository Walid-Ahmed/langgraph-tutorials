# Minimal counter: one integer in state, one node adds 1 each invoke.
# Same graph, shown twice — without a checkpointer and with MemorySaver.
#
# Run from the repository root:
#   python "7-Checkpointing/02-memory-saver/00_simple_counter.py"

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    count: int


def increment(state: State):
    # Read the current count and return an update that adds 1.
    return {"count": state["count"] + 1}


# Build the same one-node graph for both demos:
# START -> increment -> END
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)


# ---------------------------------------------------------------------------
# Part 1 — WITHOUT a checkpointer
# ---------------------------------------------------------------------------
# Each invoke() is independent. Nothing from the previous run is saved.
graph_no_memory = builder.compile()

print("=== Without checkpointer ===")
for i in range(3):
    # We must pass a starting count every time — there is no saved state.
    result = graph_no_memory.invoke({"count": 0})
    print(f"invoke {i + 1}: count = {result['count']}")
# Always prints 1, 1, 1 — each run starts from 0 and adds 1.


# ---------------------------------------------------------------------------
# Part 2 — WITH MemorySaver + a fixed thread_id
# ---------------------------------------------------------------------------
# The checkpointer saves the final count after each invoke().
# The next invoke() on the same thread_id restores that saved value first.
checkpointer = MemorySaver()
graph_with_memory = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "counter-demo"}}

print("\n=== With MemorySaver ===")
for i in range(3):
    # First call: seed the counter at 0.
    # Later calls: pass {} so we do NOT overwrite the restored count.
    # (Plain fields like int are replaced, not merged, when you pass them.)
    input_state = {"count": 0} if i == 0 else {}
    result = graph_with_memory.invoke(input_state, config)
    print(f"invoke {i + 1}: count = {result['count']}")
# Prints 1, 2, 3 — each run continues from the previous saved value.
