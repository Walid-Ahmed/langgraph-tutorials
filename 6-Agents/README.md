# 6. Agents

Agents are LangGraph graphs where the LLM controls more of the path at runtime.

In a workflow, you usually hardcode the main path. Even if there is a conditional edge, the graph is choosing among branches you already defined.

In an agent, the model can keep deciding what to do next. A common tool-calling agent loop is:

```text
llm_call -> should_continue -> tool_node -> llm_call
```

The loop repeats until the model stops requesting tool calls.

## Tutorials

| File | Concept | Purpose |
|---|---|---|
| `00_tool_calling_agent.md` + `00_tool_calling_agent.py` | Tool-calling agent | Let the model decide when to call tools and when to stop |

## How This Relates To Workflows

`5-Workflows/00_augmented_llm.md` explains the building blocks: tools, structured output, retrieval, and memory.

This folder shows what happens when tool use becomes an agent loop: the model can call tools repeatedly until it has enough information to answer.
