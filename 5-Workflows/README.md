# 5. Workflows

This folder teaches common LangGraph workflow patterns: how an LLM system can call tools, chain steps, route work, run tasks in parallel, delegate to workers, and improve its own output.

This folder mixes runnable examples with concise concept pages. The runnable examples come first, and the remaining pages mark patterns that will get code later.

Reference: [LangGraph Workflows and Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents#llms-and-augmentations)

Source figures: [Workflow types deck](https://docs.google.com/presentation/d/1SuUtO_OCcglql3DRU5KnbKtxxHokxPhX/edit?slide=id.p2#slide=id.p2)

![Workflow roadmap](figures/workflow-00-roadmap.png)

## Part 1 — Core Tutorial

A workflow is a reusable pattern for organizing how an LLM system thinks, calls tools, makes decisions, and checks results. The official LangGraph docs describe workflows as more predictable than open-ended agents: the path is mostly defined by code, even when an LLM helps with a step.

Earlier folders teach the building blocks:

- state
- nodes
- reducers
- message history
- conditional edges

This folder uses those building blocks to explain larger workflow patterns. The goal is not to memorize names; it is to recognize the shape of the problem and pick the simplest graph that fits.


## Visual Workflow Map

| Workflow | Figure | Core Idea |
|---|---|---|
| Augmented LLM | <img src="figures/workflow-01-augmented-llm.png" alt="Augmented LLM" width="220"> | Add tools, structured output, or memory to the model |
| Prompt chaining | <img src="figures/workflow-02-prompt-chaining.png" alt="Prompt chaining" width="220"> | Pass one LLM result into the next step |
| Routing | <img src="figures/workflow-03-routing.png" alt="Routing" width="220"> | Classify input and send it to the right path |
| Parallelization | <img src="figures/workflow-04-parallelization.png" alt="Parallelization" width="220"> | Run independent LLM calls at the same time |
| Orchestrator-workers | <img src="figures/workflow-05-orchestrator-workers.png" alt="Orchestrator workers" width="220"> | Let a manager break work into subtasks |
| Evaluator-optimizer | <img src="figures/workflow-06-evaluator-optimizer.png" alt="Evaluator optimizer" width="220"> | Generate, evaluate, improve, and stop when good enough |

## Workflow Tutorials

| File | Workflow | Purpose |
|---|---|---|
| `00_augmented_llm.md` + `00_augmented_llm.py` | Augmented LLM with tools | LLM enhanced with tools, retrieval, or memory |
| `00_augmented_llm_structured_output.md` + `00_augmented_llm_structured_output.py` | Augmented LLM with structured output | LLM output constrained by a Pydantic schema |
| `01_prompt_chaining.md` + `01_prompt_chaining.py` + `01_prompt_chaining_joke_gate.py` | Prompt chaining | Break tasks into ordered LLM steps, with an optional quality gate |
| `02_routing.md` | Routing | Send work to different paths based on input or state |
| `03_parallelization.md` + `03_parallelization.py` + `03_parallelization_creative.py` | Parallelization | Run independent LLM calls side by side, then combine them |
| `04_orchestrator_workers.md` | Orchestrator-workers | One controller delegates work to specialized workers |
| `05_evaluator_optimizer.md` | Evaluator-optimizer | Generate, evaluate, and improve outputs iteratively |

## Supporting Resources

| Resource | Purpose |
|---|---|
| `resources/langchain_augmentation_snippets.md` | Small LangChain snippets for structured output and tool binding before they are used inside LangGraph workflows |

## How To Use This Folder

Start with the runnable examples:

1. `00_augmented_llm.py` for tools and `ToolNode`
2. `00_augmented_llm_structured_output.py` for Pydantic output
3. `01_prompt_chaining.py` for a sequential content pipeline
4. `01_prompt_chaining_joke_gate.py` for prompt chaining with a quality gate
5. `03_parallelization_creative.py` for a minimal joke/story/poem fan-out
6. `03_parallelization.py` for a practical social-media content package

Then read the concept pages for routing, orchestrator-workers, and evaluator-optimizer. Those pages explain the pattern before code is added.

Each tutorial should make clear:

1. what state fields are needed
2. which nodes are involved
3. how edges connect the workflow
4. where conditional edges or reducers are used
5. what output to expect
