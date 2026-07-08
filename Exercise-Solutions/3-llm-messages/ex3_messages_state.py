# Exercise 3 — Extend MessagesState with turn_count
#
# MessagesState already provides a messages field with add_messages.
# We subclass it to add turn_count and increment it on every chatbot call.

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class MyChatState(MessagesState):
    turn_count: int


def chatbot_node(state: MyChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "turn_count": state["turn_count"] + 1,
    }


def main() -> None:
    graph = StateGraph(MyChatState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    app = graph.compile()
    plot_graph(app)

    final_state = app.invoke({
        "messages": [HumanMessage(content="What is RAG?")],
        "turn_count": 0,
    })

    print(f"turn_count: {final_state['turn_count']}")  # Expected: 1
    print("Messages:")
    for msg in final_state["messages"]:
        print(f"  {msg.__class__.__name__}: {msg.content[:80]}...")


if __name__ == "__main__":
    main()
