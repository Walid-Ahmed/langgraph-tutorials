# 7.2 MemorySaver — Three Ways to Handle Conversation Memory

**Example files (run in this order):**

| File | Strategy | Remembers across invokes? |
|---|---|---|
| [`00_no_memory.py`](00_no_memory.py) | no checkpointer (the baseline) | no |
| [`01_memory_saver.py`](01_memory_saver.py) | `MemorySaver` + `thread_id` | yes (same process) |
| [`02_manual_history.py`](02_manual_history.py) | the caller threads history itself | yes (caller's job) |

**Requires:** `OPENAI_API_KEY` in the repo-root `.env` — all three run the same tiny chatbot graph (`START → chat → END`) so the *only* thing that differs is how memory is handled.

Tutorial 3 built a chatbot whose history accumulated *within* one `invoke()`, then vanished when it returned. This mini-section fixes that — three times over — so you can see exactly which component owns the transcript in each approach. The contrast is the lesson: the same graph behaves completely differently depending on who remembers.

## The Concept: Memory Is Something You Attach, Not Something Graphs Have

**What is it?** By default a compiled graph is stateless between runs: each `invoke()` starts from the input you pass and forgets everything when it returns. A **checkpointer** changes that — attach one at compile time and pass a `thread_id`, and LangGraph reloads the previous run's state before the new input arrives, so the conversation continues instead of restarting.

**`MemorySaver`, specifically:** it's the simplest checkpointer, storing snapshots in the running Python process's memory.

```text
MemorySaver
→ same process : remembers
→ process exits: forgets
```

That volatility is a *teaching* feature, not a flaw. It has zero setup — no database, no config — so it isolates the checkpointing *API* from the mechanics of durable storage. Once the API is clear, swapping in a durable backend ([`08-postgres-saver/`](../08-postgres-saver/README.md)) is a one-line change.

**Intuition:** `MemorySaver` is a conversation held entirely in someone's head. Perfect recall while the meeting is in session; the moment they walk out (the process exits), it's gone. Writing it down in a shared notebook — that's `PostgresSaver`, later.

## The Three Strategies, Side by Side

**`00_no_memory.py` — the baseline.** No checkpointer. Run 1: "Hi, my name is Walid." Run 2: "What is my name?" → *"I don't know your name."* Each `invoke()` starts blank. This is the problem the other two solve, and the reason memory needs to be *added*.

**`01_memory_saver.py` — the checkpointer.** The same chatbot, plus three lines: a `MemorySaver`, `compile(checkpointer=...)`, and a `thread_id` in the config. Run 2 on thread `"walid-session"` → *"Your name is Walid!"* The caller passed only the new question; LangGraph restored the earlier turn from the checkpoint, and `add_messages` appended the new one.

**`02_manual_history.py` — do it yourself.** No checkpointer at all — the *caller* carries the transcript forward and passes the whole thing back in each time:

```python
result = graph.invoke({"messages": result["messages"] + [new_user_turn]})
```

This also works, and it's a legitimate pattern — it's exactly what tutorial 6's Exercise 3 had you do by hand. Comparing it against the checkpointer is the whole point of this section:

| | No memory (`00`) | Checkpointer (`01`) | Manual history (`02`) |
|---|---|---|---|
| Remembers across invokes | no | yes | yes |
| Who owns the transcript | nobody | LangGraph, keyed by thread | your calling code |
| Caller passes per turn | new message | new message + `thread_id` | *entire* history + new message |
| Multiple concurrent users | n/a | trivial — one thread each | you build the bookkeeping |
| Also unlocks crash-resume & pauses | no | **yes** | no |

The last row is the decider. Manual history buys you *memory only*. The checkpointer's real dividend — crash recovery and human-in-the-loop pauses — is what the rest of tutorial 7 is about.

## Run Them

From the repo root, in order:

```bash
python "7-Checkpointing/02-memory-saver/00_no_memory.py"     # "I don't know your name."
python "7-Checkpointing/02-memory-saver/01_memory_saver.py"  # "Your name is Walid!"
python "7-Checkpointing/02-memory-saver/02_manual_history.py" # also remembers — caller carries it
```

What to notice in each:

- **`00`** — the second invoke has no idea who you are; nothing carried over.
- **`01`** — the same `thread_id` is what lets LangGraph reload the previous turn; the script also prints the stored state so you can see what the checkpointer kept.
- **`02`** — no checkpointer is involved; the memory lives entirely in the caller's variable.

## Key Takeaways

1. A graph is **stateless between runs by default** — memory is a component you add, not a built-in behavior.
2. `MemorySaver` + `thread_id` gives automatic conversational memory: the caller passes only the new turn, and LangGraph restores the rest.
3. Manual history is a valid alternative for *memory alone*, but the caller owns all the bookkeeping and gets none of the resume/pause features.
4. `MemorySaver` is **not durable** — it dies with the process. It teaches the API; production swaps in a database-backed saver with no change to the graph.

## Next in This Tutorial

The [tutorial 7 overview](../README.md) builds on this into crash recovery and human-in-the-loop pauses, then [`08-postgres-saver/`](../08-postgres-saver/README.md) makes the memory survive a process restart.
