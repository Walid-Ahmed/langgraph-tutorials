# PostgreSQL Setup for LangGraph

**Book mapping:** Appendix F — *PostgreSQL for LangGraph: Install, Connect, Verify.*

This folder is not a tutorial step. It is the one-time setup behind every PostgreSQL example in the repo:

- [`7-Checkpointing/06-postgres-saver/`](../7-Checkpointing/06-postgres-saver/) — durable checkpoints with `PostgresSaver` (book Chapter 9)
- [`8-Long-Term-Memory/03-postgres-store/`](../8-Long-Term-Memory/03-postgres-store/) — durable long-term memory with `PostgresStore` (book Chapter 10)

Do it once, on the machine you develop on. If you already have a server, a database and a working `DB_URI`, skip to step 4.

## Files

| Step | File | Book | What it does |
|---|---|---|---|
| 1 | this README | Listings F.1–F.4 | Install PostgreSQL and confirm the server runs |
| 2 | [`01-create-database.sql`](01-create-database.sql) | Listings F.5–F.6 | Create a login role `langgraph` and a database `langgraph_stm` it owns |
| 3 | [`02-crud-demo.sql`](02-crud-demo.sql) | Listing F.7 | Optional: a `notes` table and `INSERT` / `SELECT` / `UPDATE` / `DELETE` by hand |
| 4 | [`.env.example`](.env.example) | Listing F.8 | The connection string, as `DB_URI` |
| 5 | [`03-verify_connection.py`](03-verify_connection.py) | Listing F.9 | Prove the connection from Python with `psycopg` (no LangGraph) |
| 6 | [`04-setup_langgraph_tables.py`](04-setup_langgraph_tables.py) | Listing F.10 | Create the `PostgresSaver` and `PostgresStore` tables, once |
| 7 | [`05-pooled_saver.py`](05-pooled_saver.py) | Listing F.11 | A connection pool for a long-running service |

## 1. Install PostgreSQL

Pick one route.

**macOS — Postgres.app.** Download from postgresapp.com, open it and click **Initialize**. Then put its tools on your `PATH` (Listing F.1):

```bash
echo 'export PATH="/Applications/Postgres.app/Contents/Versions/latest/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
psql --version
```

**macOS — Homebrew** (Listing F.2):

```bash
brew install postgresql@17
brew services start postgresql@17

# versioned formulae are keg-only, so add them to PATH explicitly
echo 'export PATH="$(brew --prefix)/opt/postgresql@17/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
psql --version
```

**Windows.** Run the EDB installer from postgresql.org/download/windows. Write down the `postgres` superuser password, keep port `5432`, and keep pgAdmin 4 ticked.

**Debian / Ubuntu** (Listing F.3):

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
systemctl status postgresql        # should report active
sudo -u postgres psql              # connect as the superuser
```

**Fedora / RHEL** (Listing F.4):

```bash
sudo dnf install postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl enable --now postgresql
sudo -u postgres psql
```

Confirm the *server* is listening, not just that the client is installed:

```bash
psql -h localhost -p 5432 -U postgres -d postgres -c "SELECT version();"
```

On Postgres.app and Homebrew the superuser is your macOS username, so drop `-U postgres`.

## 2. Create the role and the database

Run as a superuser (macOS shown; the file's header has the Linux and Windows commands):

```bash
psql -h localhost -d postgres -f 13-Postgres-Setup/01-create-database.sql
```

```text
CREATE ROLE
CREATE DATABASE
```

## 3. Optional: a table of your own

```bash
psql -h localhost -p 5432 -U langgraph -d langgraph_stm -f 13-Postgres-Setup/02-crud-demo.sql
```

This is also the first real test of the new role's password. Drop the table afterwards with `DROP TABLE notes;`.

## 4. Put the connection string in `.env`

```bash
cp 13-Postgres-Setup/.env.example 13-Postgres-Setup/.env
```

or add the same `DB_URI=...` line to the repo-root `.env`. The scripts here call `load_dotenv()`, which looks in this folder first and then up to the repo root. The `7-Checkpointing` and `8-Long-Term-Memory` Postgres examples read `DB_URI` from the repo-root `.env`, so that is the simplest single place for it.

If you skipped step 2 and use your own role (common on a Mac), the string is just `postgresql://<your-username>@localhost:5432/langgraph_stm?sslmode=disable`.

## 5–7. Verify from Python, create LangGraph's tables, try a pool

Run from the repo root, with the virtual environment active:

```bash
python 13-Postgres-Setup/03-verify_connection.py
```

```text
connected to : langgraph_stm
as role      : langgraph
server       : PostgreSQL 17.4 ...
```

```bash
python 13-Postgres-Setup/04-setup_langgraph_tables.py
```

```text
checkpointer tables ready
store tables ready
```

```bash
python 13-Postgres-Setup/05-pooled_saver.py
```

```text
pool ok: SELECT 1 returned 1 (max 20 connections)
checkpointer ready: PostgresSaver
```

The packages (`psycopg[binary,pool]` and `langgraph-checkpoint-postgres`) are already in the repo's `requirements.txt`.

Now the Chapter 9 and Chapter 10 Postgres examples will run on your machine.

## When it fails

| Error | Usually means | Book |
|---|---|---|
| `Connection refused`, or `database "langgraph_stm" does not exist` for a database you created | Two servers on one port; you reached the other one. `lsof -i :5432` | F.9, port conflicts |
| `Peer authentication failed` / `no pg_hba.conf entry` | Connect over TCP with `-h localhost`; the URI already does | F.9, authentication |
| `database "..." does not exist` | List what this server has: `psql -h localhost -U postgres -l` | F.9 |
| `KeyError: 'DB_URI'` | No `.env` with `DB_URI` was found (step 4) | F.7 |
| `sorry, too many clients already` | One connection per request; use a pool (`05-pooled_saver.py`) | F.9, connection exhaustion |

Note: this folder's `13-` prefix comes from the book's Appendix F and is unrelated to `13-Agent-Server/`.
