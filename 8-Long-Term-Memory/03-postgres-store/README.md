# Durable Long-Term Memory with `PostgresStore`

The earlier examples use `InMemoryStore`. It shares user facts across different
`thread_id` values, but its contents disappear when Python stops.

This example replaces that temporary backend with `PostgresStore`:

```text
process 1: put(namespace, key, profile) → PostgreSQL → process stops
process 2: get(namespace, key)          ← PostgreSQL → profile is restored
```

No LLM is used. This keeps the lesson focused on storage and makes the
demonstration deterministic and free to run.

## `PostgresStore` Is Not `PostgresSaver`

Both can use the same PostgreSQL server or database, but they store different
information in different tables:

| Component | Saves | Identity |
|---|---|---|
| `PostgresSaver` | messages, graph state, and execution position | `thread_id` |
| `PostgresStore` | selected user or application memories | namespace + key |

```text
PostgresSaver
└── thread_id = "chat-1"
    └── this conversation's checkpoints

PostgresStore
└── namespace = ("memory", "user-1")
    └── key = "user_details"
        └── {"name": "Walid", "role": "..."}
```

The same user can start `chat-1`, `chat-2`, and `chat-3`. Those conversations
remain separate in the checkpointer, while all three can read the profile under
the same Store namespace.

## Files

| File | Purpose |
|---|---|
| `00_setup_tables.py` | create or migrate the PostgresStore tables |
| `01_save_profile.py` | save one structured profile and exit |
| `02_read_profile.py` | start a new process and retrieve that profile |
| `run_demo.sh` | run all three scripts in order |

## 1. Install Dependencies

From the repository root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The relevant packages are:

```text
langgraph-checkpoint-postgres
psycopg[binary,pool]
```

Despite its package name, `langgraph-checkpoint-postgres` supplies both
`PostgresSaver` and `PostgresStore`.

## 2. Start PostgreSQL and Create a Database

For a Homebrew PostgreSQL 16 installation:

```bash
brew services start postgresql@16
createdb -h localhost -p 5432 langgraph_stm
```

`createdb` is needed only once. If the database already exists, continue.

The Python scripts do not install PostgreSQL, start the server, or create the
database. They only create LangGraph tables inside an existing database.

## 3. Configure `.env`

Add the connection string to the repository-root `.env` file:

```text
DB_URI=postgresql://walidahmed@localhost:5432/langgraph_stm?sslmode=disable
```

Change the username, password, host, port, or database for your environment.
This example does not require `OPENAI_API_KEY`.

## 4. Run the Three Processes

Run each command from the repository root:

```bash
python "8-Long-Term-Memory/03-postgres-store/00_setup_tables.py"
python "8-Long-Term-Memory/03-postgres-store/01_save_profile.py"
python "8-Long-Term-Memory/03-postgres-store/02_read_profile.py"
```

The second command finishes before the third starts. Therefore, the final
profile cannot be coming from a Python variable—it was reloaded from PostgreSQL.

Alternatively, run the complete sequence:

```bash
chmod +x "8-Long-Term-Memory/03-postgres-store/run_demo.sh"
./8-Long-Term-Memory/03-postgres-store/run_demo.sh
```

## What the Code Stores

The save script uses:

```python
namespace = ("memory", "user-1")
key = "user_details"
value = {
    "name": "Walid",
    "role": "Engineering manager",
    "preferences": ["concise explanations"],
}

store.put(namespace, key, value)
```

The read script reconnects and uses the same address:

```python
item = store.get(("memory", "user-1"), "user_details")
print(item.value)
```

Using the same namespace and key in `put()` again updates the existing entry.
Using another `user_id` creates an isolated user-memory namespace.

## One-Time Setup

Before using a new database, run:

```python
with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()
```

`setup()` creates and migrates the Store schema. Run it during deployment,
application initialization, or a database migration step—not before every
memory operation.

## What This Proves

- Long-term scope: the profile is addressed by `user_id`, not `thread_id`.
- Durable storage: the profile survives after the writer process exits.
- Structured value: Store saves the profile as a dictionary, not chat history.
- Separate responsibilities: Store memory is independent of checkpoint memory.
