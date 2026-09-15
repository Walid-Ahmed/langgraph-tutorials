# 5. Run Until Interrupt

This is the smallest planned-interrupt example in the repo.

The graph has two nodes:

```text
START -> step_one -> pause before step_two -> step_two -> END
```

The graph is compiled like this:

```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["step_two"],
)
```

That means the first `invoke()` runs `step_one`, saves a checkpoint, and stops
before `step_two`.

```python
result = graph.invoke({"log": []}, config)
```

## Inspecting the Paused State

Inspecting the paused state is useful, but it is not what makes resume work.
The checkpoint already exists after the first `invoke()`.

This line reads the frozen snapshot from the checkpointer:

```python
snapshot = graph.get_state(config)
```

Use `get_state()` when your app needs to:

- confirm the graph really paused;
- see the next node waiting to run;
- show saved values to a user before continuing.

In this example, the saved snapshot says:

```python
snapshot.values  # {"log": ["step_one"]}
snapshot.next    # ("step_two",)
```

`snapshot.next` is the clearest signal that the graph is paused before
`step_two`. `snapshot.values` is the state the next node will receive.

Then this resumes from the same checkpoint:

```python
final_state = graph.invoke(None, config)
```

`None` means there is no new input. LangGraph should load the paused checkpoint
identified by the same `thread_id` and continue at `step_two`.

You could skip `get_state()` and still resume:

```python
result = graph.invoke({"log": []}, config)
final_state = graph.invoke(None, config)
```

That works because resume depends on the saved checkpoint and the same
`thread_id`, not on whether you inspected the checkpoint.

## Run

From the repository root:

```bash
python "7-Checkpointing/05-run-until-interrupt/00_run_until_interrupt.py"
```

Expected shape:

```text
=== Invoke #1: run until interrupt ===
step_one ran
Returned state: {"log": ["step_one"]}
Next node to run: ("step_two",)

=== Invoke #2: resume from interrupt ===
step_two ran
Final state: {"log": ["step_one", "step_two"]}
```

## Key Lesson

`interrupt_before` is a planned pause. It is useful when the next step should
wait for something outside the graph, such as a human decision, a scheduled
time, or an external system check.

The three practical phases are:

```text
1. Run until interrupt
2. Review paused state, if the app or user needs it
3. Inject updates if needed, then resume
```
