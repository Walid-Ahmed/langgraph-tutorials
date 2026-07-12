# State Snapshots and Reducers

This mini-section shows what a checkpointer actually stores and how restored state merges with new input.

| File | What it proves |
|---|---|
| [`00_custom_state_reducer.py`](00_custom_state_reducer.py) | fields without reducers are overwritten; fields with reducers accumulate |

## Why This Belongs in Checkpointing

Checkpointing is not only "remembering messages." It restores the previous graph state, then merges the new input using the state's update rules.

```text
restored checkpoint state
+ new invoke input
+ reducer rules
= next state
```

That is why reducers matter here:

```text
field without reducer → overwritten
field with reducer    → accumulated/combined
```

## Run

From the repo root:

```bash
python "7-Checkpointing/01-state-snapshots/00_custom_state_reducer.py"
```
