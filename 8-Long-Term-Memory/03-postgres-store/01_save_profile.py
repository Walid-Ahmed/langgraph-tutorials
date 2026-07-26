# Process 1 of the PostgresStore durability demonstration.
#
# This script writes one structured user profile and then exits. The next
# script starts a fresh Python process and proves the profile survived.
#
# Run from the repository root after 00_setup_tables.py:
#   python "8-Long-Term-Memory/03-postgres-store/01_save_profile.py"

import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.store.postgres import PostgresStore

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DB_URI = os.getenv("DB_URI")

# namespace identifies the user's memory space. key identifies one item inside
# that space. Neither one is a thread_id, so this memory can be shared by many
# conversations belonging to the same user.
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
    profile = {
        "name": "Walid",
        "role": "Engineering manager",
        "preferences": ["concise explanations"],
    }

    with PostgresStore.from_conn_string(require_db_uri()) as store:
        # put() saves a JSON-like dictionary under namespace + key. Calling
        # put() again with the same namespace and key updates this item.
        store.put(NAMESPACE, KEY, profile)

        saved_item = store.get(NAMESPACE, KEY)

    assert saved_item is not None
    print("Saved durable long-term memory:")
    print(f"  namespace: {saved_item.namespace}")
    print(f"  key:       {saved_item.key}")
    print(f"  value:     {saved_item.value}")
    print("\nThis Python process can now stop.")
    print(
        'Next: python "8-Long-Term-Memory/03-postgres-store/'
        '02_read_profile.py"'
    )


if __name__ == "__main__":
    main()
