# MemorySaver Comparison

This mini-section compares three ways to handle conversation history.

| File | What it proves |
|---|---|
| [`00_no_memory.py`](00_no_memory.py) | without a checkpointer, each `invoke()` starts fresh |
| [`01_memory_saver.py`](01_memory_saver.py) | `MemorySaver` remembers across invokes in the same Python process |
| [`02_manual_history.py`](02_manual_history.py) | the caller can manually pass the full message history instead |

## Core Idea

`MemorySaver` is a LangGraph checkpointer that stores state in Python memory:

```text
MemorySaver
→ same process: remembers
→ process stops: forgets
```

It teaches the checkpointing API, but it is not durable. That is why the PostgresSaver example later uses PostgreSQL.

## Run

From the repo root:

```bash
python "7-Checkpointing/02-memory-saver/00_no_memory.py"
python "7-Checkpointing/02-memory-saver/01_memory_saver.py"
python "7-Checkpointing/02-memory-saver/02_manual_history.py"
```

## What to Notice

- `00_no_memory.py`: the second invoke does not know the name.
- `01_memory_saver.py`: the same `thread_id` lets LangGraph reload the previous turn.
- `02_manual_history.py`: no checkpointer is used; the caller carries the messages manually.
