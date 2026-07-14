# 7.8 PostgreSQL Checkpointing — Memory That Survives a Restart

**Example files (run in this order):**

| File | Role |
|---|---|
| [`00_setup_tables.py`](00_setup_tables.py) | one-time: create/validate the checkpoint tables |
| [`01_save_name.py`](01_save_name.py) | process A: save the first turn, then exit |
| [`02_recall_name.py`](02_recall_name.py) | process B: a *new* process recalls the name from PostgreSQL |

**Requires:** `OPENAI_API_KEY`, a running PostgreSQL server, an existing database, and a `DB_URI` in the repo-root `.env`. This is the one section in the series that needs infrastructure outside Python — that is precisely the point.

Every checkpointing example before this one used `MemorySaver`, which stores snapshots in the running Python process's memory. It taught the API perfectly, but it has one fatal limitation for real software: **when the process exits, the memory is gone.** This section swaps that one component for a database-backed saver and proves the difference by splitting a single conversation across two separate Python processes — the first writes, exits, and the second, started fresh, still remembers.

## The Concept: Same API, Durable Storage

**What is it?** `PostgresSaver` is LangGraph's PostgreSQL-backed checkpointer. It implements the exact same interface as `MemorySaver` — you attach it at `compile()` and address threads by `thread_id` — but it writes checkpoints to PostgreSQL tables instead of to process memory. Nothing else about your graph changes.

**What problem does it solve?** Durability. A chatbot must not forget a conversation because the server restarted; a long-running workflow must not replay from step one after a deployment. The moment your application needs memory to outlive a single process, the in-memory saver is disqualified and a database-backed one takes its place.

**Why is the change so small?** Because LangGraph deliberately makes the checkpointer a pluggable component. `MemorySaver`, `SqliteSaver`, and `PostgresSaver` are interchangeable behind one interface. You learn the mechanics with the zero-setup in-memory saver, then graduate to a durable backend without touching your nodes, edges, or state schema.

**Intuition:** think of the difference between a document open in an editor with *no file on disk* versus one that has been *saved to a file*. `MemorySaver` is the unsaved buffer — rich and fast while the program runs, vanishing when you close it. `PostgresSaver` is Save-to-disk: quit the editor, reboot the machine, reopen tomorrow, and the work is still there. The editing experience is identical; only the persistence differs.

## Short-Term vs. Long-Term Memory — Read This First

A common misconception is that "persistent" means "long-term memory." It does not. LangGraph separates two scopes, and `PostgresSaver` covers only the first:

| Memory type | Scope | LangGraph tool | Holds |
|---|---|---|---|
| Short-term memory | one thread / conversation | **checkpointer** (`PostgresSaver`) | message history, current graph state, interrupt/resume position |
| Long-term memory | across threads and sessions | **store** (`PostgresStore`) | user preferences, durable facts, profile data |

`PostgresSaver` is **durable *thread* memory**. It can store many threads in one database:

```text
PostgresSaver in PostgreSQL
├── thread_id = "chat_session_walid"   → messages + state for this chat
├── thread_id = "support_ticket_123"   → state for another workflow
└── thread_id = "student_session_9"    → state for another conversation
```

But every saved memory is still keyed by `thread_id`. `PostgresSaver` never says "these three threads all belong to Walid, so share facts between them." Cross-thread facts — the kind keyed by a stable user id — belong in LangGraph's separate `Store` interface:

```text
user_id = "walid"
├── name = "Walid"
├── prefers concise explanations
└── is learning LangGraph
```

That is why these scripts are named `01_save_name.py` / `02_recall_name.py`, not `long_term_memory.py`: they prove *durability of a thread*, not *sharing across threads*. Rule of thumb — **checkpointer = this conversation's memory; store = this user's memory.**

## The Graph Is Not the Point

The graph here is intentionally trivial — a single chatbot node, the same shape as tutorial 3:

![PostgresSaver chatbot graph](../diagrams/08_postgres_saver_graph.png)

Both scripts compile this identical graph. The *only* thing that matters is the checkpointer attached at compile time. Keeping the graph boring is deliberate: it isolates persistence as the single variable under study.

## Who Sets Up What: You vs. the Code

`PostgresSaver` does **not** create a PostgreSQL server or a database for you. Those must already exist before any script runs. Only the checkpoint *tables* are created by code.

| Layer | Created by | How |
|---|---|---|
| PostgreSQL server | **you** | install/start PostgreSQL locally, or use a hosted database |
| Database | **you** | e.g. `createdb -h localhost -p 5432 langgraph_stm` |
| Checkpoint tables | **code** | `00_setup_tables.py` calls `checkpointer.setup()` |

So this line in your repo-root `.env`:

```text
DB_URI=postgresql://walidahmed@localhost:5432/langgraph_stm?sslmode=disable
```

assumes the `langgraph_stm` database already exists. Adjust the user, host, port, and database name to match your own PostgreSQL setup. The setup script then creates the tables *inside* that database.

## Installing `PostgresSaver`

`PostgresSaver` ships in a separate integration package. If LangGraph and a PostgreSQL driver are already present, add just the checkpointer:

```bash
pip install -U langgraph-checkpoint-postgres
```

For a fresh environment, install LangGraph, the checkpointer, and `psycopg` together:

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

Before the first run against a new database, initialize the checkpoint schema:

```bash
python "7-Checkpointing/08-postgres-saver/00_setup_tables.py"
```

Inside that script, the load-bearing line is:

```python
checkpointer.setup()          # async code: await checkpointer.setup()
```

`setup()` prepares the database for checkpoint storage: it creates the required tables if they are missing and validates the schema if they already exist. In a real project, run it once during application initialization, deployment, or a migration step — **never** inside every graph invocation.

## The Minimal Shape

A PostgreSQL checkpointer is attached exactly like `MemorySaver` — at compile time — the difference being the connection to a real database:

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user@localhost:5432/langgraph_stm?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup() lives in the one-time setup script, not here
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-123"}}
    result = graph.invoke({"messages": [...]}, config)
```

A later invocation with the *same* `thread_id` — even from a different process — restores that thread from PostgreSQL. That cross-process restore is the entire lesson.

## Code Walkthrough — Why Two Scripts?

The demo is split into two scripts *on purpose*: the split across process boundaries is what makes durability visible. A single script proving memory could always be cheating with in-process state; two scripts cannot.

### Process A — save the first turn ([`01_save_name.py`](01_save_name.py))

It runs one invoke on a stable thread id, then exits:

```python
THREAD_ID = "chat_session_walid"
message   = "Hi! My name is Walid."
```

Run the one-time setup, then Process A:

```bash
python "7-Checkpointing/08-postgres-saver/00_setup_tables.py"   # once per database
python "7-Checkpointing/08-postgres-saver/01_save_name.py"      # saves the turn, then exits
```

When this process ends, a `MemorySaver` would take the conversation to the grave. `PostgresSaver` has already written it to disk.

### Process B — recall from a brand-new process ([`02_recall_name.py`](02_recall_name.py))

Started later, in a separate Python process, it asks a question that can only be answered from memory:

```python
THREAD_ID = "chat_session_walid"    # SAME thread id — this is the hinge
message   = "What's my name?"
```

```bash
python "7-Checkpointing/08-postgres-saver/02_recall_name.py"
```

Because the `thread_id` matches, LangGraph loads the earlier `MessagesState` from PostgreSQL *before* calling the model, so the model sees the original introduction and answers "Walid." Change `THREAD_ID` and you address a different (empty) conversation — proof the id, not the code, is what reconnects to saved state.

## What PostgreSQL Actually Stores

You rarely query these tables directly — `PostgresSaver` manages them — but knowing their jobs demystifies what "a checkpoint" is:

| Table | Purpose |
|---|---|
| `checkpoints` | one checkpoint record + metadata per super-step, per thread |
| `checkpoint_blobs` | larger serialized pieces of graph state, stored separately |
| `checkpoint_writes` | intermediate writes, so completed work isn't lost if a later step fails |
| `checkpoint_migrations` | tracks checkpoint-schema versions over time |

Together they are what lets LangGraph restore state, inspect history, and resume after interruptions — the same capabilities from earlier in tutorial 7, now durable. For a GUI tour of these tables, see [Viewing LangGraph Checkpoint Tables in pgAdmin](pgadmin_view_tables.md).

## Verifying It Works

The test *is* the demo — and it only convinces because a process boundary sits in the middle:

1. `00_setup_tables.py` — once, to prepare the tables.
2. `01_save_name.py` — tell the chatbot your name.
3. **Let that Python process fully exit.**
4. `02_recall_name.py` — in a new process, ask "What's my name?"
5. It answers correctly → the checkpoint survived a process restart.

Swap `PostgresSaver` back to `MemorySaver` and repeat: step 4 fails, because the memory died with process A. That contrast is the whole reason this section exists.

## Key Takeaways

1. `PostgresSaver` is the **same checkpointer API** as `MemorySaver` with durable storage — the graph code doesn't change, only where snapshots live.
2. You provide the PostgreSQL **server and database**; `checkpointer.setup()` provides the **tables**. Run setup once, not per invoke.
3. The same `thread_id` reconnects a future invoke — *even in a new process* — to saved state. That cross-process restore is what "durable" means.
4. `PostgresSaver` is durable **short-term (thread) memory**. Cross-thread, per-user facts belong in `Store` / `PostgresStore` — different tool, different scope.
5. In production, `setup()` runs at deploy/migration time; graph invocations just open a connection and go.

## Reference

- [LangGraph — add memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph — persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
