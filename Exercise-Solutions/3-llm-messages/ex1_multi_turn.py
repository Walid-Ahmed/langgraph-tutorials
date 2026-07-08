# Exercise 1 — Multi-turn conversation
#
# Start with two messages: one telling the LLM the user's name,
# then one asking for the name back. The LLM should answer "Alex"
# because it receives the full history, not just the last message.

import sys
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def main() -> None:
    graph = StateGraph(ChatState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    app = graph.compile()
    plot_graph(app)

    initial_state = {
        "messages": [
            HumanMessage(content="My name is Alex."),
            HumanMessage(content="What is my name?"),
        ]
    }

    final_state = app.invoke(initial_state)

    print("Full conversation:")
    for msg in final_state["messages"]:
        print(f"  {msg.__class__.__name__}: {msg.content}")
    # The AI message should mention "Alex".


if __name__ == "__main__":
    main()
