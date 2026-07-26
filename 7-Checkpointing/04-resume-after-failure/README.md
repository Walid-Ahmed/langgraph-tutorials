# 4. Resume After Failure

This example teaches how a checkpoint lets a graph continue after a node
fails. No API key is required.

## Graph

```text
START → step_one → step_two → step_three → END
```

`step_two` is programmed to fail on its first attempt. This is intentional: it
simulates a temporary API, network, or worker error.

## First Invoke: Run Until the Failure

```python
graph.invoke({"log": []}, config)
```

The first invocation proceeds like this:

```text
step_one succeeds
→ LangGraph saves a checkpoint
→ step_two raises RuntimeError
→ invoke() stops with an exception
```

The checkpoint records both the state and where execution should continue:

```text
state:     log = ["step_one"]
next node: step_two
thread:    resume-demo
```

The script catches the error with `try`/`except`, so the Python process remains
running.

## Second Invoke: Resume the Same Run

The script invokes the graph a second time:

```python
graph.invoke(None, config)
```

This is a resume, not a new run:

- `None` means there is no new graph input;
- the same `thread_id` selects the same checkpoint; and
- the saved next-node position tells LangGraph to continue at `step_two`.

```text
first invoke:  step_one ✓ → step_two fails
second invoke:              step_two ✓ → step_three ✓
```

`step_one` does not run again because its successful result was already
checkpointed.

## Why Step Two Succeeds the Second Time

The failure counter starts at one:

```python
attempts_before_success = {"step_two": 1}
```

The first call decreases it to zero and deliberately raises an error. When the
graph resumes, the condition is false, so `step_two` succeeds.

## Important MemorySaver Limitation

This example uses `MemorySaver`. Its checkpoints exist only in the current
Python process. It demonstrates recovery from an exception that the script
catches; it does not survive closing or restarting Python.

For recovery after a real process crash or restart, use a durable checkpointer
such as SQLite or PostgreSQL.

## Run

From the repository root:

```bash
python "7-Checkpointing/04-resume-after-failure/00_resume_after_failure.py"
```

## Key Lesson

A checkpointer saves the graph state and its next execution position. Calling
`invoke(None, config)` with the same `thread_id` restores that checkpoint and
continues from the failed step instead of restarting the graph.
