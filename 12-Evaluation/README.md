# 12. Evaluation — Golden Datasets, Offline and Online, with LangSmith

Companion code for **Chapter 14** of *Building AI Agents with LangGraph*. LangSmith is the evaluation platform throughout: datasets, experiments, feedback and annotation queues.

Every script evaluates the same small agent: the help desk of **Corniche Cycles**, a fictional bike-rental shop in Alexandria, with two tools (`rental_quote`, `bikes_available`). Its tools read fixed tables, so the right answers are known in advance.

**Files (in reading order):**
- [`desk.py`](desk.py) — the system under test, in three versions: `v1` (first draft), `v2` (fixed after the first experiment), `v3` (`v2` plus a pricing-tool refactor with an off-by-one bug). `make_llm_node()` exposes the `llm` node on its own.
- [`golden_set.jsonl`](golden_set.jsonl) — the seed for the LangSmith dataset: 12 curated examples with the expected tool calls (and arguments), facts the reply must contain, and the expected behavior in words.
- [`evaluators.py`](evaluators.py) — LangSmith evaluators: `correct_tools` (trajectory), `first_step_calls` (component: tools + arguments), `facts_present` (facts), `judge_reply` (LLM-as-judge, gpt-4o).
- [`run_experiment.py`](run_experiment.py) — uploads the dataset once, runs an experiment: `python run_experiment.py v2`
- [`component_eval.py`](component_eval.py) — evaluates the `llm` node by itself: `python component_eval.py v2`
- [`regression_gate.py`](regression_gate.py) — compares a candidate with the latest stored baseline experiment, example by example; exits 1 on any regression: `python regression_gate.py v2 v3`
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
