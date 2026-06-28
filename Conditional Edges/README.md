# 4. Conditional Edges

This folder shows how a graph can choose different paths based on state.

## What This Covers

- Creating a router function
- Using `add_conditional_edges()`
- Sending the graph to different nodes based on a score
- Ending both branches cleanly

## File

| File | Purpose |
|---|---|
| `05_conditional_edges.py` | Grades an answer, then routes to `pass_node` or `retry_node` |

## Graph Plot

```mermaid
flowchart TD
    START([START]) --> GRADE["grade_node"]
    GRADE -. "score >= 70" .-> PASS["pass_node"]
    GRADE -. "score < 70" .-> RETRY["retry_node"]
    PASS --> END([END])
    RETRY --> END
```

Solid edges always run. Dotted edges are conditional and are chosen by the router function.

## Key Idea

A router is not a normal node. It reads the current state and returns the name of the next node.

## Run

```bash
python "Conditional Edges/05_conditional_edges.py"
```
