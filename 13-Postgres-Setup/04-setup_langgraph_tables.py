# Run from the repository root, ONCE (not on every application start):
#   python 13-Postgres-Setup/04-setup_langgraph_tables.py
#
# What it does:
#   Creates the tables PostgresSaver (checkpoints, Chapter 9) and
#   PostgresStore (long-term memory, Chapter 10) need, in the same database.
#
# What it demonstrates (Appendix F, §F.8, Listing F.10):
#   - LangGraph's Postgres backends do not create tables on import; .setup() does
#   - .setup() is safe to run again: it creates what is missing and migrates
#     its own schema between LangGraph versions
#
# Prerequisites:
#   03-verify_connection.py succeeds (same DB_URI).
#
# Expected output:
#   checkpointer tables ready
#   store tables ready
#
# Afterwards, `\dt` in psql (or Tables in pgAdmin) shows LangGraph's tables in
# the public schema, next to the notes table from 02-crud-demo.sql.

import os
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

load_dotenv()
DB_URI = os.environ["DB_URI"]

# Both adapters point at the same database and create their own tables.
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    print("checkpointer tables ready")

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()
    print("store tables ready")
