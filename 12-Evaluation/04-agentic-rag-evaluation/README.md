# Evaluating an agentic RAG chatbot with LangSmith

Companion code for **Chapter 14** of *Building AI Agents with LangGraph*. This is the Chapter 12 chatbot from [`9-Agentic-RAG`](../../9-Agentic-RAG/README.md) with its evaluation added. The application and its evaluation design come from Kerolos Yacoub's practical session ([Evaluation_practical](https://github.com/Kerolos2019/Evaluation_practical)), ported from DeepEval and Langfuse to **LangSmith**.

An internal chatbot answers questions about company data (CRM exports, projects, revenue and proposal PDFs). It's **agentic RAG**: retrieval is a tool, `search_knowledge_base`, that the model decides to call. It can also skip it, as it does for the weather tool's questions. Classic RAG would retrieve before every answer.

The application is evaluated twice:

| Stage | When | What runs | Where results go |
|---|---|---|---|
| Offline evaluation | Before release, in CI | `evals/run_eval.py`: 18 golden cases × 9 metrics, pass rule, 80% gate, baseline comparison | A LangSmith experiment, plus `evals/results/latest.json` |
| Online evaluation | After release, on every live answer | `src/agents.py`, `_score_online`: toxicity, bias, answer relevancy, without a golden answer | LangSmith feedback on the answer's trace |

## Layout

```text
../../9-Agentic-RAG/datasets/  company files (CSV + PDF), shared with the Chapter 12 app
src/                      the chatbot
  config.py               paths, models, chunk size
  loaders.py              CSV / PDF / TXT -> text
  chunking.py             1000-character chunks, 200 overlap
  vectorstore.py          Chroma collection with OpenAI embeddings
  pipeline.py             ingestion: load -> chunk -> store (runs once)
  tools.py                search_knowledge_base, get_weather (live, Open-Meteo)
  agents.py               as in 03, plus run_id, retrieved sources and online scores
  app.py                  Streamlit chat UI
evals/                    the evaluation harness
  data/golden_dataset.json  18 golden cases
  metrics/                rule-based, heuristic, safety, RAG and LLM-as-judge metrics
  evaluators.py           the nine metrics as LangSmith evaluators
  gate.py                 pass rule, summary, baseline comparison
  run_eval.py             the offline evaluation run and CI gate
  results/latest.json     per-case scores + summary of the last run
online_traffic.py         simulated live sessions, scored online; reads the feedback back
review_queue.py           flagged runs + a random sample into a LangSmith annotation queue
promote_case.py           a reviewed production question into the golden set
tests/                    unit tests for the metrics, the gate and the review selection (no API keys)
```

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Then, in this folder, copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `LANGSMITH_API_KEY`.

## Run

From `12-Evaluation/04-agentic-rag-evaluation/`:

```bash
python -m src.pipeline                    # ingest datasets/ into Chroma (once)
python evals/run_eval.py                  # offline evaluation + CI gate
python evals/run_eval.py --set-baseline   # keep a reviewed run as the baseline
streamlit run src/app.py                  # the chat UI, with online scoring
python online_traffic.py                  # simulated live traffic, scored online
python review_queue.py                    # queue flagged and sampled runs for review
python promote_case.py                    # add the reviewed case to the golden set
```

Unit tests, from the repository root:

```bash
python -m pytest -q 12-Evaluation/04-agentic-rag-evaluation/tests
```

**Results we got** (`gpt-4o-mini` agent, `gpt-4o` judge; Chapter 14, Section 14.10, tells the story):

```text
run 1  17/18 (94%)   snowflake_sales_rep answers "REP01"; the judge passes won_opportunities at 0.8
                     although it lists a lost deal as won
run 2  16/18 (89%)   stricter judge rubric: won_opportunities now fails, as it should
run 3  18/18 (100%)  query rule in the search tool's docstring + row-check rule in the prompt; baseline
run 4  17/18 (94%)   n_results 4 -> 2: REGRESSED graphrag_project_owner, exit 1 (change reverted)
```

**Pass rule:** a case passes when `correctness >= 0.8` and the agent's first tool is the expected one. The run fails (exit 1) when fewer than 80% of cases pass, or when a case that passed in `evals/results/baseline.json` fails now.

## Changes from the original

- LangSmith replaces Langfuse (tracing, sessions, scores) and DeepEval (the two GEval judges and answer relevancy). The judges are plain LangChain calls with a fixed rubric and structured output.
- The RAG metrics grade the chunks the agent's own `search_knowledge_base` calls returned, not a separate retrieval with the raw question.
- The safety keyword lists match whole words, so "skill" no longer counts as "kill" and "call" no longer counts as "all".
- Golden case `accounts_north_america` used the key `expected_tools`; it's now `expected_tool`, like the others, and its "HeathFirst" typo is now "HealthFirst".
- The company data in `9-Agentic-RAG/datasets/` is new and synthetic (`make_datasets.py`), written so all 18 golden answers hold; the original data wasn't published.
- The agent runs on `gpt-4o-mini` and the judge on `gpt-4o`, as in the rest of the book (the original used `gpt-5`).
- `get_weather` returns live weather from [Open-Meteo](https://open-meteo.com) (free for non-commercial use, no API key) instead of dummy text. The `london_weather` golden case therefore describes the expected behavior rather than a temperature, and the correctness judge's rubric says how to grade such cases.
