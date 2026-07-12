# Part 1 of the PostgresSaver restart demo.
#
# This script saves the first conversation turn to PostgreSQL:
# "Hi! My name is Walid."
#
# Why this is useful:
# - With MemorySaver, memory disappears when the Python process stops.
# - With PostgresSaver, this message is saved in PostgreSQL under THREAD_ID.
# - After this script exits, run 02_recall_name.py to prove the
#   next process can still remember the name.
#
# How to run this example:
# 1. Start PostgreSQL if needed:
#    brew services start postgresql@16
# 2. Create the database once if it does not exist:
#    createdb -h localhost -p 5432 langgraph_stm
# 3. Put these values in the repo root .env file:
#    OPENAI_API_KEY=your_key_here
#    DB_URI=postgresql://walidahmed@localhost:5432/langgraph_stm?sslmode=disable
# 4. Create/validate LangGraph checkpoint tables once:
#    python "7-Checkpointing/08-postgres-saver/00_setup_tables.py"
# 5. Save the first turn:
#    python "7-Checkpointing/08-postgres-saver/01_save_name.py"
# 6. Then run the second script in a separate process:
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
        # Tables should already exist. Create them once with:
        # python "7-Checkpointing/08-postgres-saver/00_setup_tables.py"
        graph = build_graph(checkpointer)
        plot_graph(graph, GRAPH_PATH)

        # Same THREAD_ID must be used by 02_recall_name.py.
        # That is how the second script loads this saved conversation.
        config = {"configurable": {"thread_id": THREAD_ID}}

        message = "Hi! My name is Walid."
        result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)

        print(f"\nUser: {message}")
        print(f"AI: {result['messages'][-1].content}")
        print("\nSaved this turn in PostgreSQL.")
        print("Now stop this script and run: python \"7-Checkpointing/08-postgres-saver/02_recall_name.py\"")


if __name__ == "__main__":
    main()
