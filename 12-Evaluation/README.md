# 12. Evaluation — Golden Datasets, Offline and Online, with LangSmith

Companion code for **Chapter 14** of *Building AI Agents with LangGraph*. LangSmith is the evaluation platform throughout: datasets, experiments, feedback and annotation queues.

Every script evaluates the same small agent: the help desk of **Corniche Cycles**, a fictional bike-rental shop in Alexandria, with two tools (`rental_quote`, `bikes_available`). Its tools read fixed tables, so the right answers are known in advance.

**Files (in reading order):**
- [`desk.py`](desk.py) — the system under test, in three versions: `v1` (first draft), `v2` (fixed after the first experiment), `v3` (`v2` plus a pricing-tool refactor with an off-by-one bug). `make_llm_node()` exposes the `llm` node on its own.
- [`golden_set.jsonl`](golden_set.jsonl) — the seed for the LangSmith dataset: 12 curated examples with the expected tool calls (and arguments), facts the reply must contain, and the expected behavior in words.
- [`evaluators.py`](evaluators.py) — LangSmith evaluators: `correct_tools` (trajectory), `first_step_calls` (component: tools + arguments), `facts_present` (facts), `judge_reply` (LLM-as-judge, gpt-4o).
- [`run_experiment.py`](run_experiment.py) — uploads the dataset once, runs an experiment: `python run_experiment.py v2`
- [`component_eval.py`](component_eval.py) — evaluates the `llm` node by itself: `python component_eval.py v2`
- [`regression_gate.py`](regression_gate.py) — compares a candidate with the latest stored baseline experiment on its exact dataset snapshot, example by example; exits 1 on a regression or incomplete results: `python regression_gate.py v2 v3`
- [`online_feedback.py`](online_feedback.py) — online evaluation: traced "production" runs, user thumbs and a reference-free grounding check attached as feedback.
- [`review_queue.py`](review_queue.py) — sends flagged production runs to a LangSmith annotation queue for human review.
- [`promote_example.py`](promote_example.py) — adds one reviewed production case to the dataset, linked to its source run.

**Requires:** `OPENAI_API_KEY`, `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true`. Copy [`.env.example`](.env.example) to `.env` in this folder. Run every script from this folder, in the order above.

**Cost:** one experiment is about 25 gpt-4o-mini calls and 12 gpt-4o judge calls, a few US cents.

**Results we got** (answers are worded differently on each run; these scores were the same on every run):

```text
v1: correct_tools 0.83 | facts_present 0.83 | judge_reply 0.83
v2: correct_tools 1.00 | facts_present 1.00 | judge_reply 1.00
regression_gate.py v2 v3 -> REGRESSED price-boundary (facts_present, judge_reply 1.00 -> 0.00), exit 1
```

## Inputs, state and scores

For each dataset example, LangSmith passes `example.inputs` to the evaluation target. The target calls `run_desk`, which invokes the graph and returns `answer`, `tools_called` and `tool_results` from the final state. These become the target's `run.outputs`. Evaluators receive those actual outputs and, when requested, the example's reference outputs. The reference answers aren't passed into this graph. Evaluators can request `inputs`, `outputs`, `reference_outputs`, or the full `run` and `example` objects.

An evaluator approach says how you grade (code, an LLM judge or a human); a metric says what you measure (correctness, safety, latency); a metric value is the result (1, pass, 230 ms). One experiment can track ten or more metrics. Operational values such as latency and cost can come from traces.

## A complete comparison on a fixed dataset

The gate reads the baseline's `dataset_version`, fetches that historical set with `as_of`, and runs the candidate on the same inputs and references. It rejects missing examples, missing/non-finite scores, failed runs and repeated example results. It expects one run per example; aggregate repetitions before using a different comparison policy.

For CI, pin the exact experiment you reviewed:

```bash
python regression_gate.py v2 v3 --baseline-experiment desk-v2-YOUR_SUFFIX
```

Replace the name with your stored experiment's name. Without this option, the gate selects the latest `desk-v2-*` experiment. If the dataset changes, run and review a new baseline to include the new or revised cases. An old pinned snapshot deliberately excludes later changes. Wait for baseline runs and feedback to finish ingesting before using the gate; incomplete ingestion produces FAIL. Configure repository rules to require the CI check if its failure must block merging.

Exercise 14.3 deliberately selects only `correct_tools` for both sides with `gate("v2", "v3", metrics=("correct_tools",), evaluators=(correct_tools,))`. That narrow gate cannot detect the price bug. Merely deleting candidate metrics while leaving the default three required metrics now fails.

## Local tests with pytest

From the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest -q 12-Evaluation/test_pricing.py
python -m pytest -q 12-Evaluation/test_regression_gate.py
```

pytest is the testing framework; the unit test is the check it runs. `test_pricing.py` tests the real tool at 4, 5 and 6 hours, then applies `facts_present` to both the correct and deliberately broken tool outputs. `test_regression_gate.py` uses fake LangSmith responses to check coverage, errors and snapshot handling. These tests make no model or LangSmith API calls. The judge is created only when `judge_reply` runs, so code-only evaluators need no OpenAI key.
