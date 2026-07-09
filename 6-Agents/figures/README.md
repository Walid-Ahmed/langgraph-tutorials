# Agent Figures

Concept illustration embedded in the [agent tutorial](../README.md).

Unlike the auto-generated plot in [`diagrams/`](../diagrams/README.md), this is a **static teaching figure** — a fixed picture of the idea, not something a script regenerates. It's shown near the top of the tutorial to fix the shape of the tool-calling loop in the reader's mind before the code arrives.

| File | Illustrates |
|---|---|
| `agent-tool-loop.png` | The tool-calling agent loop: `llm → (needs a tool?) → tools → llm → … → END` |

For the same loop shown *executing over time* — messages accumulating across iterations — see the sequence diagram in the [tutorial's execution walkthrough](../README.md#execution-walkthrough).
