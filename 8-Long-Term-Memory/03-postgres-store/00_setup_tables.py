# One-time database setup for the PostgresStore tutorial.
#
# PostgresStore stores long-term memories shared across conversations. Its
# tables are different from the checkpoint tables created by PostgresSaver.
#
# Before running:
# 1. Start PostgreSQL.
# 2. Create the database if needed.
# 3. Add DB_URI to the repository-root .env file.
#
# Run from the repository root:
#   python "8-Long-Term-Memory/03-postgres-store/00_setup_tables.py"

import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DB_URI = os.getenv("DB_URI")


def require_db_uri() -> str:
    """Return DB_URI or stop with a beginner-friendly setup message."""
    if not DB_URI:
        raise SystemExit(
            "Missing DB_URI.\n"
            f"Add it to {REPO_ROOT / '.env'}, for example:\n"
            "DB_URI=postgresql://walidahmed@localhost:5432/"
            "langgraph_stm?sslmode=disable"
        )
    return DB_URI


def main() -> None:
    # setup() creates or migrates the tables used by PostgresStore. It does
    # not create the PostgreSQL server or the database itself.
    with PostgresStore.from_conn_string(require_db_uri()) as store:
        store.setup()

    print("PostgresStore tables are ready.")
    print(
        'Next: python "8-Long-Term-Memory/03-postgres-store/'
        '01_save_profile.py"'
    )


if __name__ == "__main__":
    main()
