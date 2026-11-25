# Run from the repository root:
#   python "Exercise-Solutions/3-llm-messages/ex3_messages_state.py"
#
# Exercise 3.6 — Extend MessagesState (Appendix B, Listing B.7)
#
# What it shows: MessagesState already gives you `messages` with the
# add_messages reducer. We subclass it and add turn_count with its OWN
# reducer (operator.add), so the node returns 1 and the reducer adds it.
# Two keys, two reducers: messages appends, turn_count adds.
# Needs OPENAI_API_KEY in the repository-root .env file.
#
# Expected output:
#   2 1     (two messages: the question and the reply; one turn)

import sys
from operator import add
from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class ChatState(MessagesState):          # inherits messages + add_messages
    turn_count: Annotated[int, add]


def chat(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response], "turn_count": 1}     # "+1", merged by the reducer


def main() -> None:
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    app = graph.compile()
    plot_graph(app, print_mermaid=False)

    result = app.invoke({"messages": [HumanMessage("Hello!")], "turn_count": 0})
    print(len(result["messages"]), result["turn_count"])


if __name__ == "__main__":
    main()
