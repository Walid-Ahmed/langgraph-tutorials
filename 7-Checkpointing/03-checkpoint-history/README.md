# 3. Checkpoint History

You already learned this evaluator-optimizer style of loop in `5-Workflows`.
The loop is reused here only as context for a new checkpointing skill:
inspecting the saved execution history after the graph finishes.

The graph follows this path:

```text
intake → analyze → (revise → analyze)* → finalize
```

An LLM returns a structured quality score, issues, and recommendation. The
router revises weak content until it reaches the acceptance threshold or
`MAX_ITERATIONS` prevents an endless loop. The checkpointing lesson begins
after execution, when `get_state_history()` prints the snapshot saved after
each node.

## Run

Add `OPENAI_API_KEY` to the repository-root `.env`, then run from the
repository root:

```bash
python "7-Checkpointing/03-checkpoint-history/00_document_review_loop.py"
```

## Key lesson

Checkpointing is not only conversational memory. It can make a looping
workflow inspectable by recording how its state and routing decisions changed
over time.
