# Evaluating an agentic RAG chatbot with LangSmith

Companion code for **Chapter 14** of *Building AI Agents with LangGraph*. The application and its evaluation design come from Kerolos Yacoub's practical session ([Evaluation_practical](https://github.com/Kerolos2019/Evaluation_practical)), ported from DeepEval and Langfuse to **LangSmith**.

An internal chatbot answers questions about company data (CRM exports, projects, revenue and proposal PDFs). It's **agentic RAG**: retrieval is a tool, `search_knowledge_base`, that the model decides to call. It can also skip it, as it does for the weather tool's questions. Classic RAG would retrieve before every answer.

The application is evaluated twice:

| Stage | When | What runs | Where results go |
|---|---|---|---|
| Offline evaluation | Before release, in CI | `evals/run_eval.py`: 18 golden cases × 9 metrics, pass rule, 80% gate, baseline comparison | A LangSmith experiment, plus `evals/results/latest.json` |
| Online evaluation | After release, on every live answer | `src/agents.py`, `_score_online`: toxicity, bias, answer relevancy, without a golden answer | LangSmith feedback on the answer's trace |

## Layout

```text
datasets/                 company files the chatbot answers from (CSV + PDF)
src/                      the chatbot
  config.py               paths, models, chunk size
  loaders.py              CSV / PDF / TXT -> text
  chunking.py             1000-character chunks, 200 overlap
  vectorstore.py          Chroma collection with OpenAI embeddings
  pipeline.py             ingestion: load -> chunk -> store (runs once)
  tools.py                search_knowledge_base, get_weather (dummy)
  agents.py               create_agent, LangSmith tracing, online scores
  app.py                  Streamlit chat UI
evals/                    the evaluation harness
  data/golden_dataset.json  18 golden cases
  metrics/                rule-based, heuristic, safety, RAG and LLM-as-judge metrics
  evaluators.py           the nine metrics as LangSmith evaluators
  gate.py                 pass rule, summary, baseline comparison
  run_eval.py             the offline evaluation run and CI gate
  results/latest.json     per-case scores + summary of the last run
tests/                    unit tests for the metrics and the gate (no API keys)
```

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Then, in this folder, copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `LANGSMITH_API_KEY`.

## Run

From `12-Evaluation/03-agentic-rag-evaluation/`:

```bash
python -m src.pipeline                    # ingest datasets/ into Chroma (once)
python evals/run_eval.py                  # offline evaluation + CI gate
python evals/run_eval.py --set-baseline   # keep a reviewed run as the baseline
streamlit run src/app.py                  # the chat UI, with online scoring
```

Unit tests, from the repository root:

```bash
python -m pytest -q 12-Evaluation/03-agentic-rag-evaluation/tests
```

**Pass rule:** a case passes when `correctness >= 0.8` and the agent's first tool is the expected one. The run fails (exit 1) when fewer than 80% of cases pass, or when a case that passed in `evals/results/baseline.json` fails now.

## Changes from the original

- LangSmith replaces Langfuse (tracing, sessions, scores) and DeepEval (the two GEval judges and answer relevancy). The judges are plain LangChain calls with a fixed rubric and structured output.
- The RAG metrics grade the chunks the agent's own `search_knowledge_base` calls returned, not a separate retrieval with the raw question.
- The safety keyword lists match whole words, so "skill" no longer counts as "kill" and "call" no longer counts as "all".
- Golden case `accounts_north_america` used the key `expected_tools`; it's now `expected_tool`, like the others.
- The agent runs on `gpt-4o-mini` and the judge on `gpt-4o`, as in the rest of the book (the original used `gpt-5`).
