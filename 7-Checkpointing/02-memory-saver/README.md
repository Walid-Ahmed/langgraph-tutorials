# MemorySaver — Your First Short-Term Memory

This tutorial starts with a simple counter, then uses chat examples to show why
a normal LangGraph invocation forgets earlier runs and how a `MemorySaver`
checkpointer gives a graph short-term, thread-scoped memory.

## TL;DR

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "chat-001"}}

graph.invoke(
    {"messages": [{"role": "user", "content": "My name is Walid"}]},
    config=config,
)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config=config,
)
```

The trick is:

> **Checkpointer saves/restores state + `thread_id` selects the state + reducer
> merges the new input.**

Or, even shorter:

> **Checkpointer at compile; state and `thread_id` at invoke.**

## What You Will Learn

By the end, you should be able to explain:

- why two ordinary `graph.invoke()` calls do not share state;
- what the first argument to `graph.invoke()` contains;
- why `thread_id` belongs in the separate `config` argument;
- how `MemorySaver` restores the correct conversation;
- how the `add_messages` reducer appends a new turn to restored messages; and
- why in-memory checkpointing does not survive a Python process restart.

## The Four Examples

| File | Who carries the previous state? | Remembers? |
|---|---|---|
| [`00_simple_counter.py`](00_simple_counter.py) | compares nobody with LangGraph's checkpointer | demonstrates both behaviors without an LLM |
| [`01_no_memory.py`](01_no_memory.py) | nobody | no |
| [`02_memory_saver.py`](02_memory_saver.py) | LangGraph's checkpointer | yes, while the process runs |
| [`03_manual_history.py`](03_manual_history.py) | your application | yes, if it resends the full history |

The counter comes first because one integer makes the save-and-restore behavior
easy to see: without memory its output is `1, 1, 1`; with `MemorySaver` and one
`thread_id` its output is `1, 2, 3`. It requires no API key or LLM.

The last three scripts use the same one-node chatbot graph:

```mermaid
flowchart LR
    START([START]) --> CHAT["chat node"] --> END([END])
```

The graph shape is not what changes. The difference is how conversation state
is carried from one invocation to the next.

## Step 1 — Start with the Simple Counter

[`00_simple_counter.py`](00_simple_counter.py) runs the same increment node
with and without a checkpointer. It isolates the core idea before introducing
chat messages and reducers:

```text
Without checkpointer: 1, 1, 1
With MemorySaver:      1, 2, 3
```

The first version must pass `{"count": 0}` on every invocation, so each run
starts over. The checkpointed version passes the initial count only once.
Later invocations pass `{}` with the same `thread_id`, allowing LangGraph to
restore the previously saved count before the increment node runs.

## The Two Arguments to `graph.invoke()`

This line contains two separate kinds of information:

```python
graph.invoke(first_input, config=config)
```

### 1. `first_input` is new graph state

The first argument supplies the new data entering this invocation:

```python
first_input = {
    "messages": [
        {"role": "user", "content": "Hi, my name is Walid"},
    ]
}
```

It must match the graph's state schema:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### 2. `config` selects the saved thread

The second argument contains execution configuration rather than message data:

```python
config = {
    "configurable": {
        "thread_id": "walid-session",
    }
}
```

The checkpointer uses `thread_id` as a lookup key:

- same `thread_id` → restore and continue the same conversation;
- different `thread_id` → start or continue a separate conversation.

LangGraph does not automatically create a new `thread_id` for every
`graph.invoke()`. Your application chooses the ID and passes it in `config`.

## The Complete Memory Cycle

```text
First invoke
    new input: "My name is Walid"
        ↓
    graph runs
        ↓
    checkpointer saves the resulting state under "chat-001"

Second invoke with the same thread_id
    checkpointer restores the state saved under "chat-001"
        ↓
    add_messages merges the new question into restored messages
        ↓
    graph runs with the complete conversation
        ↓
    checkpointer saves the updated state
```

Immediately before the chat node runs for the second time, its state contains:

```text
1. User:      Hi, my name is Walid
2. Assistant: Hello, Walid!
3. User:      What is my name?
```

That complete state is why the model can answer with the user's name.

## Step 2 — See the Problem Without Memory

[`01_no_memory.py`](01_no_memory.py) compiles without a checkpointer:

```python
graph = builder.compile()
```

It then runs twice:

```python
graph.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is Walid"}]}
)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]}
)
```

The second call receives only the second input. Reusing the same compiled
`graph` Python object does not preserve the first call's state.

## Step 3 — Add `MemorySaver`

[`02_memory_saver.py`](02_memory_saver.py) makes three important changes.

### Create the checkpointer

```python
checkpointer = MemorySaver()
```

`MemorySaver` stores checkpoints in the current Python process's RAM.

### Attach it when compiling

```python
graph = builder.compile(checkpointer=checkpointer)
```

This enables LangGraph to save graph-state snapshots as the graph executes.

### Pass a thread ID when invoking

```python
config = {"configurable": {"thread_id": "walid-session"}}

graph.invoke(first_input, config=config)
graph.invoke(second_input, config=config)
```

Both calls use `walid-session`, so the second call restores the state produced
by the first call.

## Step 4 — Understand the Reducer

The checkpointer restores the old state, but LangGraph still needs a rule for
combining the restored `messages` list with the new input. That rule is the
`add_messages` reducer:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

Conceptually:

```text
restored messages + new user message → combined messages
```

Without an append-style reducer, a new value for `messages` would normally
replace the old value. The responsibilities are different:

| Component | Responsibility |
|---|---|
| `MemorySaver` | saves and restores graph state |
| `thread_id` | selects which saved state to use |
| `add_messages` | merges new messages with restored messages |

A reducer alone is not persistent memory. Without a checkpointer, its merged
state disappears after `invoke()` returns.

## Step 5 — Inspect What Was Saved

After invoking the graph, use the same config to fetch its latest checkpoint:

```python
saved_snapshot = graph.get_state(config)

print("Stored values:", saved_snapshot.values)
print("Next nodes:", saved_snapshot.next)
```

- `values` contains the saved graph-state fields, including the messages.
- `next` contains the nodes scheduled to run next.
- `next` is empty after this example reaches `END`.

## Same Thread vs New Thread

```python
walid_chat = {"configurable": {"thread_id": "chat-001"}}
new_chat = {"configurable": {"thread_id": "chat-002"}}

graph.invoke(first_input, config=walid_chat)
graph.invoke(second_input, config=walid_chat)  # restores chat-001
graph.invoke(second_input, config=new_chat)    # separate empty thread
```

One saver can contain many independent threads. A thread normally represents
one conversation or workflow, not one person. The same user may have several
conversation IDs.

## The Manual Alternative

[`03_manual_history.py`](03_manual_history.py) proves that a checkpointer is not
the only way to preserve conversational context. The caller can retain the
returned messages and send the entire history again:

```python
first_result = graph.invoke(first_input)
conversation_history = first_result["messages"]

second_input = {
    "messages": conversation_history
    + [{"role": "user", "content": "What is my name?"}]
}

second_result = graph.invoke(second_input)
```

This works, but your application becomes responsible for storing, loading, and
resending the history. A checkpointer automates state persistence and also
supports checkpoint history, interrupts, resume, and fault recovery.

## MemorySaver's Important Limitation

```text
MemorySaver
├── same Python process: remembers
└── process exits: forgets
```

`MemorySaver` is ideal for learning and tests. It is not durable storage. If
you stop the script and run it again, its earlier checkpoints are gone.

For persistence across restarts, use a database-backed checkpointer such as
SQLite or PostgreSQL. The graph-facing pattern remains the same:

```text
checkpointer at compile + thread_id at invoke
```

## Setup

Install the repository dependencies:

```bash
python -m pip install -r requirements.txt
```

The counter needs no API key. To run the three chat examples, create a
repository-root `.env` file containing your API key:

```env
OPENAI_API_KEY=your_key_here
```

Do not commit `.env` to GitHub.

## Run the Comparison

From the repository root:

```bash
python "7-Checkpointing/02-memory-saver/00_simple_counter.py"
python "7-Checkpointing/02-memory-saver/01_no_memory.py"
python "7-Checkpointing/02-memory-saver/02_memory_saver.py"
python "7-Checkpointing/02-memory-saver/03_manual_history.py"
```

What to observe:

1. `00_simple_counter.py` prints `1, 1, 1` without memory and `1, 2, 3` with it.
2. `01_no_memory.py` cannot recall the name.
3. `02_memory_saver.py` recalls it automatically through the same thread.
4. `03_manual_history.py` recalls it because the caller resends the history.

Model responses can vary, but the presence or absence of the earlier messages
is deterministic.

## Common Mistakes

### Compiling without the checkpointer

```python
graph = builder.compile()  # no persistence
```

Fix:

```python
graph = builder.compile(checkpointer=MemorySaver())
```

### Forgetting the thread ID

A checkpointed graph needs a thread key to know where to save and load state:

```python
config = {"configurable": {"thread_id": "chat-001"}}
graph.invoke(new_input, config=config)
```

### Changing the thread ID accidentally

```python
graph.invoke(first_input,  {"configurable": {"thread_id": "chat-001"}})
graph.invoke(second_input, {"configurable": {"thread_id": "chat-002"}})
```

Those calls belong to different conversations, so the second one cannot recall
the first.

### Expecting MemorySaver to survive a restart

It stores data only in RAM. Use SQLite or PostgreSQL when checkpoints must
survive after the Python process stops.

### Confusing a reducer with a checkpointer

- Reducer: merges updates during state processing.
- Checkpointer: persists state between invocations.

Conversational memory in this example needs both.

## Exercises

1. Change the second invocation to `thread_id="new-session"`. Does it still
   know the name?
2. Add a third invocation using `walid-session` and ask another follow-up.
3. Print `graph.get_state(config).values["messages"]` after each turn.
4. Stop the script, restart it, and explain why the old thread disappeared.
5. Modify the manual-history example so it deliberately drops the first turn.

## Key Takeaways

1. `graph.invoke(input, config=config)` receives new state as its first
   argument and thread configuration separately.
2. A checkpointer saves and restores graph state between invocations.
3. `thread_id` selects the conversation state to restore.
4. `add_messages` merges the new turn into the restored message list.
5. Reusing a thread continues it; changing the thread creates isolation.
6. `MemorySaver` lasts only as long as the current Python process.

Next: [`../03-checkpoint-history/`](../03-checkpoint-history/) shows how to
inspect the sequence of snapshots produced while a graph runs. For durable
memory across restarts, continue to
[`../06-postgres-saver/`](../06-postgres-saver/).

Further reading: [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence).
