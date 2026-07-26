# Viewing LangGraph Checkpoint Tables in pgAdmin

This guide shows how to use **pgAdmin 4** to inspect the PostgreSQL tables created by `PostgresSaver`.

![pgAdmin 4 welcome screen](images/pgadmin_welcome.png)

This is the pgAdmin welcome screen. From here, use **Add New Server** or right-click **Servers** to connect to your local PostgreSQL database.

## Install pgAdmin 4

On macOS with Homebrew:

```bash
brew install --cask pgadmin4
```

Open it from the terminal:

```bash
open -a pgAdmin\ 4
```

Or open it from:

```text
Applications → pgAdmin 4
```


## What You Create vs What the Code Creates

This is the most important setup detail:

| Thing | Who creates it? | Example |
|---|---|---|
| PostgreSQL server | You install/start it | local PostgreSQL on `localhost:5432` |
| PostgreSQL database | You create it manually in pgAdmin or `psql` | `langgraph_stm` |
| LangGraph checkpoint tables | `00_setup_tables.py` creates them with `checkpointer.setup()` | `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`, `checkpoint_migrations` |

So for this `DB_URI`:

```text
DB_URI=postgresql://postgres:root@localhost:5432/langgraph_stm?sslmode=disable
```

The database named `langgraph_stm` must already exist. Then `00_setup_tables.py` connects to it and `checkpointer.setup()` creates the LangGraph checkpoint tables inside it.

In short:

```text
You / pgAdmin:
1. Register/connect to the PostgreSQL server
2. Create the database, for example langgraph_stm

Python code:
1. Connect using DB_URI
2. Run `00_setup_tables.py` once
3. Create/validate the LangGraph checkpoint tables
```

## Connect to Your Local PostgreSQL Server

In pgAdmin:

1. Right-click **Servers**.
2. Select **Register → Server**.
3. In the **General** tab, set a name such as `Local PostgreSQL`.
4. In the **Connection** tab, enter your database details:

| Field | Example |
|---|---|
| Host name/address | `localhost` |
| Port | `5432` |
| Maintenance database | `langgraph_stm` |
| Username | `postgres` |
| Password | `root` |

Use the same database name, username, and password from your `DB_URI`.

For example, this URI:

```text
postgresql://postgres:root@localhost:5432/langgraph_stm?sslmode=disable
```

means:

```text
username = postgres
password = root
host     = localhost
port     = 5432
database = langgraph_stm
```

## Find the LangGraph Tables

After connecting, open:

```text
Servers
→ Local PostgreSQL
→ Databases
→ langgraph_stm
→ Schemas
→ public
→ Tables
```

After `checkpointer.setup()` runs, you should see tables like:

```text
checkpoint_blobs
checkpoint_migrations
checkpoint_writes
checkpoints
```

## Run SQL Queries

Open:

```text
Tools → Query Tool
```

Then try:

```sql
select * from checkpoints;
```

```sql
select * from checkpoint_writes;
```

To focus on the tutorial thread:

```sql
select *
from checkpoints
where thread_id = 'chat_session_walid';
```

```sql
select *
from checkpoint_writes
where thread_id = 'chat_session_walid';
```

## What You Are Looking At

| Table | What it shows |
|---|---|
| `checkpoints` | saved graph snapshots for each `thread_id` |
| `checkpoint_writes` | writes produced by nodes while the graph ran |
| `checkpoint_blobs` | larger serialized state payloads |
| `checkpoint_migrations` | schema/version tracking for the checkpointer |

Seeing rows in these tables means `PostgresSaver` is storing LangGraph checkpoint data in PostgreSQL.
