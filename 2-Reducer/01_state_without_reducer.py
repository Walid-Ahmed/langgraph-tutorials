
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# Define the structure (schema) of the graph state.
# The state contains:
#   - count: an integer
#   - animals: a list of strings
class StateWithoutReducer(TypedDict):
    count: int
    animals: list[str]


# This node receives the current state and returns updates.
# Since no reducers are defined, the returned values
# completely replace the existing values for these keys.
def node_to_update(state: StateWithoutReducer) -> dict:

    # Ignore the incoming state and return new values.
    return {
        "count": 1,
        "animals": ["cat"]
    }


# Create a StateGraph using the state schema.
graph = StateGraph(StateWithoutReducer)

# Add the processing node to the graph.
graph.add_node("update_node", node_to_update)

# Define the execution flow:
# START --> update_node --> END
graph.add_edge(START, "update_node")
graph.add_edge("update_node", END)

# Compile the graph into an executable application.
app = graph.compile()

# Initial state passed to the graph.
initial_state = {
    "count": 5,
    "animals": ["lion", "tiger"]
}

# Execute the graph.
final_state = app.invoke(initial_state)

# Display the state before and after execution.
print(f"Initial State: {initial_state}")
print(f"Final State:   {final_state}")