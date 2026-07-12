# PostgreSQL Checkpointing with `PostgresSaver`

This guide explains the three-file PostgresSaver demo: [`00_setup_tables.py`](00_setup_tables.py), [`01_save_name.py`](01_save_name.py), and [`02_recall_name.py`](02_recall_name.py). The goal is simple: replace in-memory checkpointing with PostgreSQL so a graph can recover its state after a process restart, crash, or redeploy.

## Why Move Beyond `MemorySaver`?

`MemorySaver` keeps checkpoints inside the running Python process. That is perfect for learning because it requires no database, but the saved state disappears when the process exits.

For real applications, that is usually not enough. A chatbot should not forget the conversation after a restart, and a long-running workflow should not restart from step one after a deployment. A database-backed checkpointer gives the graph a durable place to store its thread state.

`PostgresSaver` is LangGraph's PostgreSQL-backed checkpointer. It lets LangGraph reload a thread later using the same `thread_id`.

## Short-Term vs Long-Term Memory

LangGraph memory has two useful scopes:

| Memory Type | Scope | LangGraph Tool | Example |
|---|---|---|---|
| Short-term memory | One thread or conversation | checkpointer | message history, current graph state, interrupt/resume position |
| Long-term memory | Across threads and sessions | store | user preferences, durable facts, profile data |

`PostgresSaver` belongs to the first category. It is durable **thread memory**: it saves checkpoints for a specific `thread_id`.

Important nuance: `PostgresSaver` can store **many threads** in the same PostgreSQL database. That means one database can hold many conversations or workflows:

```text
PostgresSaver in PostgreSQL
├── thread_id = "chat_session_walid"
│   └── messages and graph state for this chat
├── thread_id = "support_ticket_123"
│   └── state for another workflow
└── thread_id = "student_session_9"
    └── state for another conversation
```

But each saved memory is still organized by `thread_id`. `PostgresSaver` does not automatically say, "these three threads belong to Walid, so share facts between them." That is why it is **persistent thread memory**, not full long-term user memory.

Long-term memory is usually organized by a stable user or application key:

```text
user_id = "walid"
├── name = "Walid"
├── prefers concise explanations
└── is learning LangGraph
```

That cross-thread memory belongs in LangGraph's `Store` interface, such as `PostgresStore`. That is why these examples are named [`01_save_name.py`](01_save_name.py) and [`02_recall_name.py`](02_recall_name.py), not `long_term_memory.py`.


## Example Graph

The graph itself is intentionally tiny: one chatbot node. Both scripts use this same graph. The important difference is not the shape of the graph — it is the checkpointer attached when the graph is compiled.

![PostgresSaver chatbot graph](../diagrams/08_postgres_saver_graph.png)


## Setup Responsibility: You vs the Code

`PostgresSaver` does **not** create a PostgreSQL server or database for you. Those must exist before the script runs.

| Layer | Created by | Notes |
|---|---|---|
| PostgreSQL server | you | install/start PostgreSQL locally or use a hosted database |
| Database | you | create a database such as `langgraph_stm` |
| Checkpoint tables | code | `00_setup_tables.py` runs `checkpointer.setup()` |

So this value in `.env`:

```text
DB_URI=postgresql://postgres:root@localhost:5432/langgraph_stm?sslmode=disable
```

assumes the `langgraph_stm` database already exists. The setup script then creates the tables inside that database when you run it.

## Installing and Setting Up `PostgresSaver`

`PostgresSaver` lives in a separate integration package. If your project already has LangGraph and PostgreSQL driver support installed, add:

```bash
pip install -U langgraph-checkpoint-postgres
```

For a fresh environment, install LangGraph, the PostgreSQL checkpointer, and `psycopg` support together:

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

Before the first run against a new database, initialize the checkpoint schema. In this repo, that is done by:

```bash
python "7-Checkpointing/00_setup_tables.py"
```

Inside that script, the important line is:

```python
checkpointer.setup()
```

For async code, use:

```python
await checkpointer.setup()
```

`setup()` prepares the PostgreSQL database for LangGraph checkpoint storage. It creates the required tables if they are missing and checks the existing schema if they already exist. In a real project, run setup during application initialization, deployment setup, or a migration step — not inside every graph invocation.

## Minimal Shape

A PostgreSQL checkpointer is attached the same way as `MemorySaver`: at compile time.

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup() belongs in the one-time setup script
    graph = builder.compile(checkpointer=checkpointer)
```

Then use a stable `thread_id` when invoking the graph:

```python
config = {"configurable": {"thread_id": "user-123"}}

result = graph.invoke({"messages": [...]}, config)
```

Later, another invocation with the same `thread_id` can restore that thread from PostgreSQL.

## Code Walkthrough

This example is split into two scripts on purpose.

### Script A — save the first turn

[`01_save_name.py`](01_save_name.py) runs one invoke and then exits:

```python
message = "Hi! My name is Walid."
```

It saves that conversation under a stable thread id:

```python
THREAD_ID = "chat_session_walid"
```

First, create/validate the tables once:

```bash
python "7-Checkpointing/00_setup_tables.py"
```

Then run Script A normally:

```bash
python "7-Checkpointing/01_save_name.py"
```

### Script B — recall from a new process

[`02_recall_name.py`](02_recall_name.py) starts later, in a separate Python process, and asks:

```python
message = "What's my name?"
```

Because it uses the same `THREAD_ID`, LangGraph loads the earlier `MessagesState` from PostgreSQL before calling the model.

```bash
python "7-Checkpointing/02_recall_name.py"
```

This two-script structure is the proof that `PostgresSaver` is different from `MemorySaver`: the first Python process can end, but the second process still remembers because the checkpoint lives in PostgreSQL.

## What PostgreSQL Stores

You normally do not query these tables directly. `PostgresSaver` manages them for LangGraph. Conceptually, they support four jobs:

| Table | Purpose |
|---|---|
| `checkpoints` | stores checkpoint records and metadata for each thread |
| `checkpoint_blobs` | stores larger serialized pieces of graph state separately |
| `checkpoint_writes` | records intermediate writes so completed work is not lost if a later step fails |
| `checkpoint_migrations` | tracks checkpoint schema versions over time |

This layout lets LangGraph restore state, inspect history, and resume workflows after interruptions or failures.

## How to Test the Example Later

A good test for PostgreSQL checkpointing is:

1. Run `00_setup_tables.py` once to prepare the tables.
2. Run `01_save_name.py` to tell the chatbot your name.
3. Let that Python process finish.
4. Run `02_recall_name.py` in a new Python process.
5. Confirm the chatbot still knows your name.

If `PostgresSaver` is working, the second script restores the previous thread state from PostgreSQL.

## Key Takeaways

- `MemorySaver` is temporary; it disappears when the Python process exits.
- `PostgresSaver` persists thread checkpoints in PostgreSQL.
- `checkpointer.setup()` initializes or validates the database schema.
- The same `thread_id` is what reconnects future invokes to saved state.
- `PostgresSaver` is durable short-term memory; use `Store` / `PostgresStore` for long-term user facts.


For a GUI walkthrough, see [Viewing LangGraph Checkpoint Tables in pgAdmin](pgadmin_view_tables.md).

## Reference

- [LangGraph memory docs](https://docs.langchain.com/oss/python/langgraph/add-memory)
