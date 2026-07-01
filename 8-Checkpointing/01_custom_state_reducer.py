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


def main() -> None:
    graph = build_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    print("=== First invoke ===")
    first_result = graph.invoke({"foo": "", "bar": []}, config)
    print(first_result)
    print("foo is overwritten by the last node; bar accumulates both node updates.")

    print("\n=== Stored checkpoint after first invoke ===")
    print(graph.get_state(config).values)

    print("\n=== Second invoke with the same thread_id ===")
    second_result = graph.invoke({"foo": "", "bar": []}, config)
    print(second_result)
    print("bar still remembers the first run because the checkpointer restored thread 1.")

    print("\n=== Stored checkpoint after second invoke ===")
    print(graph.get_state(config).values)


if __name__ == "__main__":
    main()
