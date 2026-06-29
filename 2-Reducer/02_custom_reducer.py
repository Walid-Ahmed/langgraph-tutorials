from typing import Annotated, List
from typing_extensions import TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# Custom reducer for the "count" field.
#
# Instead of replacing the existing value,
# this reducer adds the new value to it.
# ---------------------------------------------------------
def custom_increment(current: int, new: int) -> int:
    return current + new


# ---------------------------------------------------------
# State Schema
#
# Each field specifies:
#   1. Its data type.
#   2. The reducer used when updating that field.
# ---------------------------------------------------------
class StateWithCustomReducer(TypedDict):

    # Integer field that uses our custom reducer.
    #
    # Example:
    # Current = 5
    # Update  = 1
    # Result  = 6
    count: Annotated[int, custom_increment]

    # List field that uses Python's built-in add()
    # to concatenate lists.
    #
    # Example:
    # Current = ["lion", "tiger"]
    # Update  = ["cat"]
    # Result  = ["lion", "tiger", "cat"]
    animals: Annotated[List[str], add]


# ---------------------------------------------------------
# Graph Node
#
# Receives the current state and returns updates.
# LangGraph automatically applies the reducers.
# ---------------------------------------------------------
def node_to_update(state: StateWithCustomReducer) -> dict:

    print("\nNode received state:")
    print(state)

    return {
        "count": 1,
        "animals": ["cat"]
    }


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------
def main():

    # Create the graph
    graph = StateGraph(StateWithCustomReducer)

    # Add node
    graph.add_node("update_node", node_to_update)

    # Define graph flow
    # START --> update_node --> END
    graph.add_edge(START, "update_node")
    graph.add_edge("update_node", END)

    # Compile the graph
    app = graph.compile()

    # Initial state
    initial_state = {
        "count": 5,
        "animals": ["lion", "tiger"]
    }

    print("Initial State:")
    print(initial_state)

    # Execute graph
    final_state = app.invoke(initial_state)

    print("\nFinal State:")
    print(final_state)


# Execute the program
if __name__ == "__main__":
    main()
