# Run from the repository root:
#   python 13-Postgres-Setup/05-pooled_saver.py
#
# What it does:
#   Opens a psycopg connection pool once and hands it to PostgresSaver, the
#   shape a long-running service (FastAPI, Chapter 16) uses instead of one
#   from_conn_string connection per request.
#
# What it demonstrates (Appendix F, §F.9 "Connection exhaustion", Listing F.11):
#   - one pool, opened at startup, shared by every request
#   - autocommit + dict_row are what PostgresSaver expects from a connection
#   - prepare_threshold=0 turns off prepared statements, which is required
#     behind PgBouncer in transaction-pooling mode
#
# Prerequisites:
#   04-setup_langgraph_tables.py has run (same DB_URI).
#
# Expected output:
#   pool ok: SELECT 1 returned 1 (max 20 connections)
#   checkpointer ready: PostgresSaver
#
# These keyword arguments match the versions pinned in requirements.txt
# (psycopg 3.3, langgraph-checkpoint-postgres 3.1). They move between
# releases, so check them against the version you pin.

import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()
DB_URI = os.environ["DB_URI"]

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={
        "autocommit": True,
        "row_factory": dict_row,
        "prepare_threshold": 0,
    },
)

checkpointer = PostgresSaver(pool)
# graph = builder.compile(checkpointer=checkpointer)

# Borrow one connection to show the pool works, then give it back.
with pool.connection() as conn:
    one = conn.execute("SELECT 1 AS one").fetchone()["one"]   # dict_row: rows are dicts
print(f"pool ok: SELECT 1 returned {one} (max {pool.max_size} connections)")
print(f"checkpointer ready: {type(checkpointer).__name__}")

pool.close()   # a service closes the pool on shutdown (FastAPI: in the lifespan)
