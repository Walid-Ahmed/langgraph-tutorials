# Workflow Resources

Small building-block notes that the [workflow](../README.md) and [agent](../../6-Agents/README.md) tutorials lean on, pulled out so they can be learned once in isolation.

These are **not** standalone tutorials. Each covers a plain LangChain mechanism — no graph involved — so you can understand the piece on its own before seeing it wired into a `StateGraph`. Structured output and tool binding, for instance, are ordinary LangChain features; the workflow and agent tutorials assume you've met them and focus on the *orchestration* around them.

## Resources

| File | Covers | First used by |
|---|---|---|
| [`langchain_augmentation_snippets.md`](langchain_augmentation_snippets.md) | `with_structured_output` and `bind_tools` snippets in plain LangChain | Routing & structured-output patterns ([tutorial 5](../README.md)); tool-calling agents ([tutorial 6](../../6-Agents/README.md)) |

If a workflow or agent example uses a LangChain feature that isn't obvious from the graph code itself, this is the place to look it up before diving into the wiring.
