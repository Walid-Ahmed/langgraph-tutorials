# 7.1 State Snapshots — What a Checkpointer Actually Stores

**Example file:** [`00_custom_state_reducer.py`](00_custom_state_reducer.py)
No API key needed — two plain nodes, no LLM, so nothing distracts from the mechanism.

Before a checkpointer can give a chatbot memory or resume a crashed run, it has to answer a more basic question: **what does it save, and how does that saved state combine with the next input?** This mini-section pins that down with the smallest possible experiment, so the powerful features later in tutorial 7 rest on something you've actually watched happen.

## The Concept: A Checkpoint Is State + Position

**What is it?** A checkpoint is a full snapshot of the graph's state taken after every super-step (every node run), together with the node that was *about to run next*. Attach an `InMemorySaver` and pass a `thread_id`, and LangGraph files one snapshot per step under that thread.

**Why start here?** Checkpointing is easy to mistake for "remembering messages." It's more precise than that: on a resumed thread, LangGraph **restores the previous state, then merges your new input through the state's reducers.** Miss that detail and the resume behavior looks like magic — or worse, like a bug. This example makes the merge visible by giving one field a reducer and one field none.

```text
restored checkpoint state
      + new invoke input
      + each field's reducer rule
      = next state
```

**Intuition:** a checkpoint is a saved game that records both your inventory (the state values) *and* where you're standing on the map (`next`). Loading it doesn't just restore items — it drops you back at the exact spot to continue from.

## The Experiment

The state deliberately mixes both update semantics from [tutorial 2](../../2-Reducer/README.md):

```python
class State(TypedDict):
    foo: str                            # no reducer  → overwritten
    bar: Annotated[list[str], add]      # reducer     → accumulates
```

Two nodes run in sequence (`node_a → node_b`), each writing both fields. Now invoke the **same `thread_id` twice** with the same fresh-looking input `{"foo": "", "bar": []}`:

```text
after invoke #1:  {'foo': 'b', 'bar': ['a', 'b']}
after invoke #2:  {'foo': 'b', 'bar': ['a', 'b', 'a', 'b']}
```

This is the subtlest point in all of tutorial 7. The second invoke did **not** reset the thread — your input was *merged into the restored state through the reducers*:

- `bar` **doubles** because the restored `['a', 'b']` is still there, and `add` appends the new run's `['a', 'b']` on top of it.
- `foo` only *looks* unchanged because it's overwritten every run anyway — its lack of a reducer hides the restore that `bar` exposes.

Checkpointing and reducers are **one system**, not two.

## Reading the History

The script also prints `get_state_history(config)` — one `StateSnapshot` per super-step, each recording the values *and* the pending `next` node:

```text
checkpoint 0 (next=('__start__',)): {'bar': []}
checkpoint 1 (next=('node_a',)):    {'foo': '', 'bar': []}
checkpoint 2 (next=('node_b',)):    {'foo': 'a', 'bar': ['a']}
checkpoint 3 (next=done):           {'foo': 'b', 'bar': ['a', 'b']}
```

That `next` field is the hinge for everything else in tutorial 7. Crash recovery, human-in-the-loop pauses, and **time travel** all reduce to the same move: *load a snapshot and continue from its `next`.* (Time travel is the ecosystem's name for the bonus trick history unlocks — because each snapshot carries its own `checkpoint_id`, you can invoke against an *older* checkpoint and re-run or fork from that earlier point, handy for debugging one bad step without replaying the whole run.)

## Run It

From the repo root:

```bash
python "7-Checkpointing/01-state-snapshots/00_custom_state_reducer.py"
```

Watch two things in the output: `bar` doubling on the second invoke (restore + reducer), and the checkpoint history growing by two entries — one per node — every run.

## Key Takeaways

1. A checkpoint stores the **state values *and* the `next` node** — position, not just data.
2. On a resumed thread, new input is **merged through the reducers**, not assigned — that's why `bar` doubles and `foo` doesn't.
3. `get_state_history` exposes one snapshot per super-step; the `next` field is what every resume, pause, and time-travel feature builds on.

## Next in This Tutorial

[`02-memory-saver/`](../02-memory-saver/README.md) puts this mechanism to work as conversational memory — comparing no checkpointer, `MemorySaver`, and caller-managed history side by side. For the full arc, see the [tutorial 7 overview](../README.md).
