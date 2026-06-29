# 03. Parallelization

## Part 1 — Core Tutorial

Parallelization runs independent tasks at the same time, then combines the results. Use it when several branches can work from the same input without waiting for each other.

![Parallelization workflow](figures/workflow-04-parallelization.png)

```mermaid
flowchart TD
    INPUT["input"] --> A["task A"]
    INPUT --> B["task B"]
    INPUT --> C["task C"]
    A --> JOIN["combine results"]
    B --> JOIN
    C --> JOIN
```

The mental model is simple:

1. one shared input enters the graph
2. multiple worker nodes run independently
3. each worker writes its own part of the state
4. one final node gathers the finished pieces

In LangGraph, this is a **fan-out / fan-in** shape:

- **fan-out**: `START` sends the same state to several nodes
- **fan-in**: the worker nodes all connect into one aggregation node

Parallelization is different from prompt chaining. In prompt chaining, each step depends on the previous step. In parallelization, the branches should be able to run without waiting for each other.

## When To Use

Use this pattern when several tasks do not depend on each other.

Good examples:

- analyze the same document from multiple angles
- generate several candidate answers
- run independent checks before a final response
- create different versions of the same content for different channels

Avoid this pattern when the steps must happen in a strict order. If task B needs task A's output, use prompt chaining instead.

## Part 2 — Code Examples That Reinforce The Concept

### Example A — Joke, Story, And Poem

Start here. This is the smallest version of the pattern.

The graph receives one topic, then runs three independent LLM calls:

- `generate_joke` writes `joke`
- `generate_story` writes `story`
- `generate_poem` writes `poem`
- `aggregator` combines all three into `combined_output`

Generated LangGraph plot from the code:

![Creative parallelization graph](diagrams/03_parallelization_creative_graph.png)

Run it:

```bash
python 5-Workflows/03_parallelization_creative.py
```

The fan-out happens here:

```python
parallel_builder.add_edge(START, "generate_joke")
parallel_builder.add_edge(START, "generate_story")
parallel_builder.add_edge(START, "generate_poem")
```

The fan-in happens here:

```python
parallel_builder.add_edge("generate_joke", "aggregator")
parallel_builder.add_edge("generate_story", "aggregator")
parallel_builder.add_edge("generate_poem", "aggregator")
```

### Example B — Social Media Content Package

The second example uses the same graph shape for a more practical task. It generates platform-specific content from one topic:

- Instagram post
- Twitter/X post
- LinkedIn post
- final combined package

Generated LangGraph plot from the code:

![Parallelization graph](diagrams/03_parallelization_graph.png)

Run it:

```bash
python 5-Workflows/03_parallelization.py
```

The graph starts with one topic, sends it to three platform-specific LLM nodes, then joins their outputs in one aggregator.

## Code Explanation

The creative example state has one shared input and one output field for each branch:

```python
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str
```

Each worker returns a partial state update:

```python
return {"joke": msg.content}
```

This does not overwrite the whole state. It only updates `joke`; the other fields stay available.

This example does **not** need a reducer because each parallel node writes to a different key. There is no conflict:

- `generate_joke` writes `joke`
- `generate_story` writes `story`
- `generate_poem` writes `poem`

You would need a reducer if multiple parallel nodes wrote to the same field, for example if every node returned `{"outputs": [...]}` and you wanted LangGraph to merge all lists together.

The aggregator reads the completed branch outputs and creates one final result:

```python
def aggregator(state: State) -> dict:
    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}
```

The social media example uses the same idea with different field names: `instagram_post`, `twitter_post`, and `linkedin_post`.

So the key lesson is simple: use parallelization when branches are independent, and join them only when the graph has enough information to build the final answer.
