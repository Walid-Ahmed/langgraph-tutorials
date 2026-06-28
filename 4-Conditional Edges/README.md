# 4. Conditional Edges

This tutorial teaches how a LangGraph workflow can choose different paths.

## What You'll Learn

After this tutorial, you will be able to:

- Distinguish normal edges from conditional edges
- Write a router function that returns the next node name
- Wire branching paths with `add_conditional_edges`

## Part 1 — Concept

A normal edge always goes to the same next node.

A conditional edge asks a router function where to go next.

Think of it like a small traffic controller inside the graph:

```mermaid
flowchart TD
    A["current state"] --> B["router function"]
    B --> C{"Which path?"}
    C --> D["path A"]
    C --> E["path B"]
```

In this example, the graph grades an answer. If the score is high enough, it goes to `pass_node`. If not, it goes to `retry_node`.

```mermaid
flowchart TD
    START([START]) --> GRADE["grade_node"]
    GRADE -. "score >= 70" .-> PASS["pass_node"]
    GRADE -. "score < 70" .-> RETRY["retry_node"]
    PASS --> END([END])
    RETRY --> END
```

Solid edges always run. Dotted edges are conditional.

### Normal Edge vs Conditional Edge

| Edge Type | What It Does | Example |
|---|---|---|
| Normal edge | Always goes to one next node | `START -> grade_node` |
| Conditional edge | Router chooses the next node | `grade_node -> pass_node` or `retry_node` |

## Part 2 — Code Illustration

File:

```text
05_conditional_edges.py
```

The graph starts with an answer:

```python
{
    "answer": "RAG means retrieval augmented generation.",
    "score": 0,
    "result": ""
}
```

The `grade_node` checks the answer and sets a score.

Then the router reads the score:

```mermaid
flowchart TD
    A["grade_node sets score"] --> B["route_after_grade reads score"]
    B --> C{"score >= 70?"}
    C -->|yes| D["pass_node"]
    C -->|no| E["retry_node"]
```

If the score is `70` or higher, the graph returns:

```python
"Passed ✅"
```

If the score is below `70`, the graph returns:

```python
"Retry needed 🔁"
```

Run it from the repo root:

```bash
python "4-Conditional Edges/05_conditional_edges.py"
```

### Graph Visualization

Like tutorial 1, this example prints a Mermaid diagram and saves `graph.png` in your current directory so you can see both branches before running the graph.

### Try It Yourself

Open `05_conditional_edges.py` and change the `answer` field to something that does not contain `"rag"`, for example:

```python
"answer": "I am not sure.",
```

Run the script again. The score will be `50`, the graph will route to `retry_node`, and the result will be `"Retry needed 🔁"`.

## Code Explanation

```python
def grade_node(state: AgentState) -> dict:
    if "rag" in state["answer"].lower():
        score = 90
    else:
        score = 50
    return {"score": score}
```

This node reads the answer and returns a score update.

```python
def route_after_grade(state: AgentState) -> str:
    if state["score"] >= 70:
        return "pass_node"
    return "retry_node"
```

This is the router. It is not a normal node that updates state. It returns the name of the next node.

```python
graph.add_conditional_edges(
    "grade_node",
    route_after_grade,
    {
        "pass_node": "pass_node",
        "retry_node": "retry_node",
    }
)
```

This tells LangGraph: after `grade_node`, call `route_after_grade`, then follow the matching path.

```python
graph.add_edge("pass_node", END)
graph.add_edge("retry_node", END)
```

Both possible branches end the graph cleanly.

## What You Learned

- Normal edges always go to the same next node
- Router functions **read state** and return a **node name**, not a state update
- `add_conditional_edges` connects a source node to multiple possible destinations

## Previous Step

This is the last tutorial in the series. Review earlier concepts in [1. LangGraph Basics](../1-Langgraph%20basics/README.md), [2. Reducers](../2-Reducer/README.md), or [3. LLM Messages](../3_LLM_Messages/README.md).
