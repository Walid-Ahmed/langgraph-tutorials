# Workflow Diagrams

Auto-generated graph plots for the runnable examples in [`5-Workflows/`](../README.md).

Each image is produced by the shared [`plot_graph()`](../../util.py) helper (via LangGraph's `get_graph().draw_mermaid_png()`) the moment you run the matching script — so they are **build artifacts, not hand-drawn art**. Don't edit them by hand; to refresh one, re-run its source example.

These show the **compiled topology** — the nodes and edges LangGraph sees after `compile()`. That makes them a useful complement to the hand-drawn Mermaid diagrams inside the [tutorial README](../README.md), which instead show *data flow* and *execution over time*. Read the two together: one is the wiring, the other is what travels through it.

| File | Source Example |
|---|---|
| `00_augmented_llm_structured_output_graph.png` | [`../00_augmented_llm_structured_output.py`](../00_augmented_llm_structured_output.py) |
| `01_prompt_chaining_graph.png` | [`../01_prompt_chaining.py`](../01_prompt_chaining.py) |
| `01_prompt_chaining_joke_gate_graph.png` | [`../01_prompt_chaining_joke_gate.py`](../01_prompt_chaining_joke_gate.py) |
| `03_parallelization_graph.png` | [`../03_parallelization.py`](../03_parallelization.py) |
| `03_parallelization_creative_graph.png` | [`../03_parallelization_creative.py`](../03_parallelization_creative.py) |
| `03_parallelization_translation_graph.png` | [`../03_parallelization_translation.py`](../03_parallelization_translation.py) |
| `04_orchestrator_workers_graph.png` | [`../04_orchestrator_workers.py`](../04_orchestrator_workers.py) |
| `04_orchestrator_workers_report_sections_graph.png` | [`../04_orchestrator_workers_report_sections.py`](../04_orchestrator_workers_report_sections.py) |

> **Reading the orchestrator-workers plots:** the static topology shows a single `research_worker` node between the orchestrator and the synthesizer — *not* one box per worker. That's expected. The fan-out to N workers happens at **runtime** through `Send`, and the number of workers is only decided when the graph runs. A compile-time diagram can't show a count that doesn't exist yet. See the [tutorial's orchestrator-workers section](../README.md#pattern-5--orchestrator-workers-04_orchestrator_workerspy) for how the dynamic dispatch actually unfolds.
