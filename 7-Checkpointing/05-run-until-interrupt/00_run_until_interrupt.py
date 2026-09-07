# Minimal planned interrupt example:
# step_one runs, LangGraph pauses before step_two, then invoke(None, config)
# resumes from the saved checkpoint.
#
# Run from the repository root:
#   python "7-Checkpointing/05-run-until-interrupt/00_run_until_interrupt.py"

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    log: list[str]


def step_one(state: State) -> dict:
    print("step_one ran")
    return {"log": state["log"] + ["step_one"]}


def step_two(state: State) -> dict:
    print("step_two ran")
    return {"log": state["log"] + ["step_two"]}


builder = StateGraph(State)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)
builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)

checkpointer = MemorySaver()

# This is the whole trick: run normally until LangGraph reaches step_two,
# then pause before step_two executes.
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["step_two"],
)

config = {"configurable": {"thread_id": "run-until-interrupt-demo"}}

print("=== Invoke #1: run until interrupt ===")
result = graph.invoke({"log": []}, config)

print(f"Returned state: {result}")

snapshot = graph.get_state(config)
print(f"Saved state: {snapshot.values}")
print(f"Next node to run: {snapshot.next}")

print("\n=== Invoke #2: resume from interrupt ===")
final_state = graph.invoke(None, config)

print(f"Final state: {final_state}")
