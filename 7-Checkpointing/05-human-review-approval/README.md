# 5. Human Review and Approval

This example pauses a checkpointed graph so a person can review an LLM-created
draft before the workflow continues.

The sequence is:

```text
invoke → create draft → pause → human reviews
       → update_state() → invoke(None, config) → finalize or revise
```

The graph is compiled with `interrupt_before=["review_decision"]`. Ordinary
Python collects the decision while the graph is paused, and `update_state()`
writes the approval and feedback into the saved checkpoint.

## Why Human-in-the-Loop Needs a Checkpointer

LangGraph's built-in interrupt and resume pattern requires a checkpointer. At
the pause, the checkpointer saves:

- the current state, including the generated draft;
- the execution position (`review_decision` is the next node);
- the thread identified by `thread_id`; and
- the approval and feedback later added through `update_state()`.

That saved information allows this call to restore the correct execution:

```python
final_state = graph.invoke(None, config)
```

Without a checkpointer, ordinary Python could still ask a person for input and
manually pass data into another function. However, LangGraph could not use its
built-in interrupt, `update_state()`, and checkpoint-resume behavior.

For this learning example, `MemorySaver` is sufficient:

```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["review_decision"],
)
```

`MemorySaver` keeps checkpoints only while the Python process is running. A
production application normally uses a durable database-backed checkpointer.

## What Happens After You Enter Feedback

Typing feedback does **not** resume the graph by itself. The script continues
through three separate Python statements:

```python
decision_update = ask_for_review_decision(result["draft"])
graph.update_state(config, decision_update)
final_state = graph.invoke(None, config)
```

1. `ask_for_review_decision(...)` waits for terminal input. If the draft is
   rejected, it returns a dictionary such as:

   ```python
   {"approved": False, "feedback": "Make it shorter"}
   ```

2. `update_state(...)` merges that decision into the checkpoint identified by
   the `thread_id`. The graph is **still paused** after this call.

3. `invoke(None, config)` resumes the paused execution. `None` means there is
   no new graph input, while the same `config` tells LangGraph which checkpoint
   to restore. Execution continues at `review_decision`.

4. The router reads the updated `approved` value. Approval goes to `finalize`;
   rejection goes to `revise`, which uses the saved feedback.

```text
enter feedback
  → save it with update_state()
  → graph remains paused
  → invoke(None, config)
  → review_decision runs
  → finalize or revise
```

## Run

Add `OPENAI_API_KEY` to the repository-root `.env`, then run from the
repository root:

```bash
python "7-Checkpointing/05-human-review-approval/00_human_review_approval.py"
```

The script is interactive and asks whether to approve the generated draft.

## Key lesson

Keep the user interface outside the graph. The graph should pause, expose its
saved state, accept a decision through `update_state()`, and resume using the
same `thread_id`.
