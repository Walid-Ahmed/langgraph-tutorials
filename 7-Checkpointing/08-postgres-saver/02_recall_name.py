# Part 2 of the PostgresSaver restart demo.
#
# Run this AFTER 01_save_name.py has finished.
# This script starts a new Python process and asks: "What's my name?"
# It uses the same THREAD_ID, so LangGraph reloads the earlier messages from
# PostgreSQL and the model can answer "Walid".
#
# This is the key difference from MemorySaver:
# - MemorySaver would lose the first turn when the previous process ended.
# - PostgresSaver restores the first turn from PostgreSQL.
#
# Normal run:
#    python "7-Checkpointing/08-postgres-saver/02_recall_name.py"

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

DB_URI = os.getenv("DB_URI")
llm = ChatOpenAI(model="gpt-4o")
THREAD_ID = "chat_session_walid"
GRAPH_PATH = "7-Checkpointing/diagrams/08_postgres_saver_graph.png"


def chatbot(state: MessagesState) -> dict:
    """Send the current message history to the LLM and append its reply."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_graph(checkpointer: PostgresSaver):
    """Build the graph and attach PostgreSQL checkpointing at compile time."""
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=checkpointer)


def require_db_uri() -> None:
    if not DB_URI:
        raise SystemExit(
            "Missing DB_URI. Add it to the repo root .env file, for example:\n"
            "DB_URI=postgresql://walidahmed@localhost:5432/langgraph_stm?sslmode=disable"
        )



def main() -> None:
    require_db_uri()

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        graph = build_graph(checkpointer)
        plot_graph(graph, GRAPH_PATH)

        # Same THREAD_ID as 01_save_name.py.
        # If you change this value, you create/load a different conversation.
        config = {"configurable": {"thread_id": THREAD_ID}}

        message = "What's my name?"
        result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)

        print(f"\nUser: {message}")
        print(f"AI: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
