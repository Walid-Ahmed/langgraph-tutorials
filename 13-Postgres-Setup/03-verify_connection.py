# Run from the repository root:
#   python 13-Postgres-Setup/03-verify_connection.py
#
# What it does:
#   Opens one connection with psycopg and asks the server which database,
#   which role and which version it is. No LangGraph involved.
#
# What it demonstrates (Appendix F, §F.8, Listing F.9):
#   - prove the connection string works BEFORE handing it to a graph, so you
#     know whether a failure is the database or the graph
#   - the outer `with` closes the connection (and commits); the inner one
#     closes the cursor
#
# Prerequisites:
#   A running PostgreSQL server, 01-create-database.sql applied, and DB_URI in
#   13-Postgres-Setup/.env or the repo-root .env (see .env.example).
#
# Expected output (your server line will differ):
#   connected to : langgraph_stm
#   as role      : langgraph
#   server       : PostgreSQL 17.4 (Ubuntu 17.4-1.pgdg22.04+1) on x86_64-pc-linux-gnu

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.environ["DB_URI"]   # KeyError naming the variable if it is missing

with psycopg.connect(DB_URI) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_database(), current_user, version();"
        )
        database, user, version = cur.fetchone()

print(f"connected to : {database}")
print(f"as role      : {user}")
print(f"server       : {version.split(',')[0]}")
