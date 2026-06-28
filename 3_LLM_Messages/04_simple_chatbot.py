from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from a local .env file.
# ChatOpenAI expects the OpenAI API key to be available in the environment.
load_dotenv()

# ---------------------------------------------------------
# Create the LLM
# ---------------------------------------------------------
llm = ChatOpenAI(
    # Use a deterministic model response so repeated runs are easier to compare.
    model="gpt-4.1",
    temperature=0
)


# ---------------------------------------------------------
# State Schema
#
# The state contains the conversation history.
# add_messages automatically appends new messages.
#
# Note:
# This custom ChatState manually recreates the same basic pattern as
# LangGraph's built-in MessagesState. MessagesState already includes a
# "messages" field with the add_messages reducer attached.
# ---------------------------------------------------------
class ChatState(TypedDict):
    # Annotated tells LangGraph to use add_messages whenever this field is updated.
    # Instead of replacing the list, each node can return only its new messages.
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------
# LLM Node
#
# Reads the conversation from the state,
# sends it to the LLM,
# returns ONLY the new AI message.
# ---------------------------------------------------------
def chatbot_node(state: ChatState) -> dict:

    print("\nCurrent Conversation:\n")

    # Inspect the current state before the model runs.
    # This shows exactly what context the LLM receives.
    for message in state["messages"]:
        print(message)

    # Send the COMPLETE conversation to the LLM
    response = llm.invoke(state["messages"])

    # Return only the new AI message.
    # add_messages will append it.
    return {
        "messages": [response]
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    # Create a graph whose state shape is defined by ChatState.
    # If you used LangGraph's built-in MessagesState instead, this could be:
    # graph = StateGraph(MessagesState)
    graph = StateGraph(ChatState)

    # Register the function that will run when the "chatbot" node is reached.
    graph.add_node("chatbot", chatbot_node)

    # Define the flow: START -> chatbot -> END.
    # This simple graph runs the chatbot node once and then exits.
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    # Compile the graph into an executable app.
    app = graph.compile()

    # User starts the conversation
    initial_state = {
        "messages": [
            HumanMessage(content="What is RAG?")
        ]
    }

    print("===== Initial State =====")
    print(initial_state)

    # Run the graph with the initial conversation state.
    # The final state includes the original human message plus the AI response.
    final_state = app.invoke(initial_state)

    print("\n===== Final Conversation =====")

    for message in final_state["messages"]:
        print(f"{message.__class__.__name__}: {message.content}")


if __name__ == "__main__":
    main()
