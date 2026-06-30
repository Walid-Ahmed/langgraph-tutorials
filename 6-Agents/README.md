# 6. Agents

Agents are LangGraph graphs where the LLM controls more of the path at runtime.

## What You'll Learn

After this section, you should be able to:

- Explain how an agent differs from a workflow
- Recognize the `llm -> tool -> llm` loop
- Understand how `ToolNode` and `add_messages` support tool-calling agents
- See what `ToolNode` does internally by reading the manual version first

## Prerequisites

- Complete [5. Workflows](../5-Workflows/README.md) first — especially `00_augmented_llm.md` on tool binding
- You should know: conditional edges, `add_messages`, and how tools are bound to an LLM
- **OpenAI API key required**: add `OPENAI_API_KEY=your_key_here` to a `.env` file in the repo root
- Optional: `TAVILY_API_KEY` for web search tool support (the example runs without it)

In a workflow, you usually hardcode the main path. Even if there is a conditional edge, the graph is choosing among branches you already defined.

In an agent, the model can keep deciding what to do next. A common tool-calling agent loop is:

```text
LLM call -> should we use a tool? -> tool node -> LLM call
```

The loop repeats until the model stops requesting tool calls.

![Agent tool loop](figures/agent-tool-loop.png)

## Tutorials

| File | Concept | Purpose |
|---|---|---|
| `00_tool_calling_agent_simple.md` + `00_tool_calling_agent_simple.py` | Tool-calling agent (manual) | Same loop with the tool node written out manually — start here |
| `01_tool_calling_agent.md` + `01_tool_calling_agent.py` | Tool-calling agent (full) | Realistic example with `ToolNode`, external APIs, and optional web search |

**Start with `00_tool_calling_agent_simple.py`** — it writes the tool node loop by hand so the mechanics are visible. Then read `01_tool_calling_agent.py` to see how `ToolNode` replaces that manual loop.

## How This Relates To Workflows

`5-Workflows/00_augmented_llm.md` explains the building blocks: tools, structured output, retrieval, and memory.

This folder shows what happens when tool use becomes an agent loop: the model can call tools repeatedly until it has enough information to answer.

## Exercises

**Exercise 1 — Add a new tool**

Open `00_tool_calling_agent.py` and add a `word_count(text: str) -> int` tool that counts the number of words in a string. Bind it alongside the existing tools and ask the agent: `"How many words are in the phrase 'the quick brown fox'?"` Verify the agent calls your tool rather than guessing.

**Exercise 2 — Cap the number of iterations**

The current agent loop can keep running until LangGraph hits its recursion limit if the model repeatedly requests tools. Add an `iteration_count: int` field to the state. Increment it in `call_llm`. Update `should_use_tools` to route to `END` if `iteration_count >= 5`, even if the model wants to keep calling tools. Test this by asking a question that requires many tool calls.

**Exercise 3 — Add conversation memory**

Modify the agent so it can hold a multi-turn conversation. After the graph finishes, capture the final `messages` list. On the next run, pass those messages as the initial state alongside a follow-up question. Verify the agent remembers context from the previous turn (e.g., ask `"What did I just ask you?"` as the second message).
