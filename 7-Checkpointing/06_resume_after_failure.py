# Demonstrates crash recovery via checkpointing: step_two raises on its
# first run, so the graph invoke() fails after step_one's checkpoint was
# already saved. Calling invoke(None, config) then resumes from that
# checkpoint — step_one is not re-run, only step_two and step_three are.

from operator import add
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    log: Annotated[list[str], add]


# Simulates a transient failure: step_two raises the first time it runs,
# then succeeds. In a real app this could be a flaky API call, a crashed
# worker process, etc. — anything that fails mid-graph.
attempts_before_success = {"step_two": 1}


def step_one(state: State) -> dict:
    print("Running step_one")
    return {"log": ["step_one"]}


def step_two(state: State) -> dict:
    if attempts_before_success["step_two"] > 0:
        attempts_before_success["step_two"] -= 1
        print("Running step_two -> simulated crash!")
        raise RuntimeError("step_two failed (simulated)")
    print("Running step_two -> succeeds this time")
    return {"log": ["step_two"]}


def step_three(state: State) -> dict:
    print("Running step_three")
    return {"log": ["step_three"]}


builder = StateGraph(State)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)
builder.add_node("step_three", step_three)

builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", "step_three")
builder.add_edge("step_three", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "resume-demo"}}

# ---------------------------------------------------------
# 1. First run: step_one succeeds and its checkpoint is saved,
#    then step_two raises before it can save a checkpoint of its own.
# ---------------------------------------------------------
print("=== Attempt 1 ===")
try:
    graph.invoke({"log": []}, config)
except RuntimeError as e:
    print(f"Graph crashed: {e}")

state = graph.get_state(config)
print(f"\nSaved log so far: {state.values['log']}")
print(f"Next node to run: {state.next}")  # ('step_two',) — step_one is NOT re-run

# ---------------------------------------------------------
# 2. Resume: invoking with `None` as input tells LangGraph to continue
#    from the last checkpoint for this thread_id instead of starting
#    over. step_one does not run again — only step_two and step_three do.
# ---------------------------------------------------------
print("\n=== Attempt 2 (resume) ===")
result = graph.invoke(None, config)
print(f"\nFinal log: {result['log']}")
# → ['step_one', 'step_two', 'step_three'] — step_one ran exactly once
