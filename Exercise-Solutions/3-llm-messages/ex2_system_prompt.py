# Run from the repository root:
#   python "Exercise-Solutions/3-llm-messages/ex2_system_prompt.py"
#
# Exercise 3.5 — Add a system prompt (Appendix B, Listing B.6)
#
# A SystemMessage at the front of the history shapes how the LLM responds.
# Here we instruct it to always answer in one sentence.

import sys
from pathlib import Path
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
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
            SystemMessage(content="You are a helpful assistant who always answers in exactly one sentence."),
            HumanMessage(content="Explain what a vector database is."),
        ]
    }

    final_state = app.invoke(initial_state)

    # Representative output: one sentence, e.g. "A vector database stores data
    # as numerical embeddings so it can quickly find items that are semantically
    # similar to a query."
    print(final_state["messages"][-1].content)


if __name__ == "__main__":
    main()
