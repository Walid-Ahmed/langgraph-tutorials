# Shows how a checkpointer's saved state interacts with reducers: with an
# InMemorySaver attached, invoking the same thread_id twice shows that a
# plain field ("foo") is overwritten each run while a reduced field ("bar")
# keeps accumulating across both runs. Also prints the full checkpoint history.

from operator import add
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]


def node_a(state: State) -> dict:
    """Plain field `foo` is overwritten; reduced field `bar` is appended."""
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State) -> dict:
    """This node overwrites foo again and appends another value to bar."""
    return {"foo": "b", "bar": ["b"]}


def build_graph():
    workflow = StateGraph(State)
    workflow.add_node(node_a)
    workflow.add_node(node_b)

    workflow.add_edge(START, "node_a")
    workflow.add_edge("node_a", "node_b")
    workflow.add_edge("node_b", END)

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def print_checkpoint_history(graph, config: RunnableConfig, label: str) -> None:
    """A checkpoint is saved after every super-step, i.e. after every node."""
    print(f"\n=== Checkpoint history — {label} ===")
    checkpoints = list(reversed(list(graph.get_state_history(config))))
    for step, snapshot in enumerate(checkpoints):
        pending = snapshot.next if snapshot.next else "done"
        print(f"  checkpoint {step} (next={pending}): {snapshot.values}")


def main() -> None:
    graph = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    print("=== First invoke ===")
    first_result = graph.invoke({"foo": "", "bar": []}, config)
    print(first_result)
    print("foo is overwritten by the last node; bar accumulates both node updates.")

    print("\n=== Stored checkpoint after first invoke ===")
    print(graph.get_state(config).values)

    print_checkpoint_history(graph, config, "one entry per node run so far")

    print("\n=== Second invoke with the same thread_id ===")
    second_result = graph.invoke({"foo": "", "bar": []}, config)
    print(second_result)
    print("bar still remembers the first run because the checkpointer restored thread 1.")

    print("\n=== Stored checkpoint after second invoke ===")
    print(graph.get_state(config).values)

    print_checkpoint_history(graph, config, "grows by two more entries (node_a, node_b) each invoke")


if __name__ == "__main__":
    main()
