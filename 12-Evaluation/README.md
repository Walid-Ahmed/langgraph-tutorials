# 12. Evaluation — Golden Datasets, Offline and Online

Companion code for **Chapter 14** of *Building AI Agents with LangGraph*.

Every script evaluates the same small agent: the help desk of **Corniche Cycles**, a fictional bike-rental shop in Alexandria, with two tools (`rental_quote`, `bikes_available`). Its tools read fixed tables, so the right answers are known in advance.

**Files (in reading order):**
- [`desk.py`](desk.py) — the system under test, in three versions: `v1` (first draft), `v2` (fixed after the first offline run), `v3` (`v2` plus a pricing-tool refactor with an off-by-one bug).
- [`golden_set.jsonl`](golden_set.jsonl) — 12 curated examples: question, expected tools, facts the reply must contain, and the expected behavior in words.
- [`evaluators.py`](evaluators.py) — three evaluators: `correct_tools` (trajectory, code), `facts_present` (result, code), `judge_reply` (LLM-as-judge, gpt-4o).
- [`run_offline_eval.py`](run_offline_eval.py) — offline evaluation, no LangSmith needed: `python run_offline_eval.py v2`
- [`compare_runs.py`](compare_runs.py) — the regression gate; exits 1 if the candidate is worse on any example: `python compare_runs.py v2 v3`
- [`langsmith_experiment.py`](langsmith_experiment.py) — the same evaluation as a LangSmith dataset + experiment: `python langsmith_experiment.py v2`
- [`online_feedback.py`](online_feedback.py) — online evaluation: traced "production" runs, user thumbs and a reference-free grounding check attached as feedback.
- [`harvest_failures.py`](harvest_failures.py) — pulls flagged production runs into `review_queue.jsonl`, the candidates for the next golden examples.

**Requires:** `OPENAI_API_KEY` (all scripts) and, for the last three, `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true`. Copy [`.env.example`](.env.example) to `.env` in this folder. Run every script from this folder.

**Cost:** one offline run is about 25 gpt-4o-mini calls and 12 gpt-4o judge calls, a few US cents.

**Results we got** (answers are worded differently on each run; these scores were the same on every run):

```text
v1: correct_tools 0.83 | facts_present 0.83 | judge_reply 0.83
v2: correct_tools 1.00 | facts_present 1.00 | judge_reply 1.00
v3: correct_tools 1.00 | facts_present 0.92 | judge_reply 0.92
```
