# Exercise 2 — Add a system prompt
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

    initial_state = {
        "messages": [
            SystemMessage(content="You are a helpful assistant who always answers in exactly one sentence."),
            HumanMessage(content="What is RAG?"),
        ]
    }

    final_state = app.invoke(initial_state)

    print("Full conversation:")
    for msg in final_state["messages"]:
        print(f"  {msg.__class__.__name__}: {msg.content}")
    # The AI reply should be noticeably shorter than without the system prompt.


if __name__ == "__main__":
    main()
