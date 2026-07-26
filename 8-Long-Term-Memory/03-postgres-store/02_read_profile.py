# Process 2 of the PostgresStore durability demonstration.
#
# Run this after 01_save_profile.py has finished. This new Python process
# reconnects to PostgreSQL and reads the same namespace + key.
#
# Run from the repository root:
#   python "8-Long-Term-Memory/03-postgres-store/02_read_profile.py"

import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DB_URI = os.getenv("DB_URI")
USER_ID = "user-1"
NAMESPACE = ("memory", USER_ID)
KEY = "user_details"


def require_db_uri() -> str:
    if not DB_URI:
        raise SystemExit(
            "Missing DB_URI.\n"
            f"Add it to {REPO_ROOT / '.env'}, for example:\n"
            "DB_URI=postgresql://walidahmed@localhost:5432/"
            "langgraph_stm?sslmode=disable"
        )
    return DB_URI


def main() -> None:
    with PostgresStore.from_conn_string(require_db_uri()) as store:
        item = store.get(NAMESPACE, KEY)

    if item is None:
        raise SystemExit(
            "No profile was found. Run 00_setup_tables.py and "
            "01_save_profile.py first."
        )

    print("Loaded long-term memory in a new Python process:")
    print(f"  namespace: {item.namespace}")
    print(f"  key:       {item.key}")
    print(f"  value:     {item.value}")
    print("\nThe profile survived because PostgresStore used PostgreSQL.")


if __name__ == "__main__":
    main()
