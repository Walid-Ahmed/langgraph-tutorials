# Demonstrates LangGraph's built-in add_messages reducer: a node returns
# just ONE new HumanMessage, and add_messages appends it to the existing
# "messages" list rather than replacing the whole conversation history.

from typing import Annotated, List
from typing_extensions import TypedDict
# HumanMessage is a class from LangChain. It represents a message sent by a human user.
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ---------------------------------------------------------
# State Schema
#
# The "messages" field stores a conversation history.
#
# add_messages is a built-in LangGraph reducer that
# appends new messages instead of replacing the list.
# ---------------------------------------------------------
class StateWithMessages(TypedDict):
    # Type: List[HumanMessage]
    # Reducer: add_messages
    # List[HumanMessage] is a type hint that means: A Python list whose elements are HumanMessage objects.
    # Existing messages are preserved.
    # New messages are appended automatically.
    messages: Annotated[List[HumanMessage], add_messages]


# ---------------------------------------------------------
# Graph Node
#
# Receives the current conversation history and returns
# one new message.
#
# The node simply returns the new message.
# LangGraph uses add_messages() to merge it into the state.
# ---------------------------------------------------------
def node_messages_reducer(state: StateWithMessages) -> dict:

    print("\nCurrent Messages:")
    for msg in state["messages"]:
        print(f"- {msg.content}")

    # Return ONE new message.
    # add_messages will append it to the existing history.
    return {
        "messages": [
            HumanMessage(content="Hello from the node!")
        ]
    }


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------
def main():

    # Create the graph
    graph = StateGraph(StateWithMessages)

    # Add node
    graph.add_node("update_node", node_messages_reducer)

    # Define graph flow
    # START --> update_node --> END
    graph.add_edge(START, "update_node")
    graph.add_edge("update_node", END)

    # Compile graph
    app = graph.compile()

    # Initial conversation history
    initial_state = {
        "messages": [
            HumanMessage(content="Initial message.")
        ]
    }

    print("Initial State:")
    print(initial_state)

    # Execute graph
    final_state = app.invoke(initial_state)

    print("\nFinal State:")

    for message in final_state["messages"]:
        print(message.content)


# Run the program
if __name__ == "__main__":
    main()
