# One-time setup script for the PostgresSaver tutorial.
#
# This script creates/validates the LangGraph checkpoint tables inside your
# existing PostgreSQL database. It does NOT create the PostgreSQL server and
# it does NOT create the database itself.
#
# Run this once after:
# 1. PostgreSQL is running:
#    brew services start postgresql@16
# 2. The database exists:
#    createdb -h localhost -p 5432 langgraph_stm
# 3. The repo root .env has DB_URI:
#    DB_URI=postgresql://walidahmed@localhost:5432/langgraph_stm?sslmode=disable
#
# Command:
#    python "7-Checkpointing/08-postgres-saver/00_setup_tables.py"
#
# After this succeeds, run:
#    python "7-Checkpointing/08-postgres-saver/01_save_name.py"
#    python "7-Checkpointing/08-postgres-saver/02_recall_name.py"

import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

DB_URI = os.getenv("DB_URI")


def main() -> None:
    if not DB_URI:
        raise SystemExit(
            "Missing DB_URI. Add it to the repo root .env file, for example:\n"
            "DB_URI=postgresql://walidahmed@localhost:5432/langgraph_stm?sslmode=disable"
        )

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()

    print("PostgresSaver checkpoint tables are ready.")
    print('Next: python "7-Checkpointing/08-postgres-saver/01_save_name.py"')


if __name__ == "__main__":
    main()
