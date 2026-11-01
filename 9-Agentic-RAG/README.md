# An agentic RAG chatbot

Companion code for **Chapter 12** of *Building AI Agents with LangGraph*, Section 12.1. This folder is the application alone; Chapter 14 evaluates it in [`12-Evaluation/04-agentic-rag-evaluation`](../12-Evaluation/04-agentic-rag-evaluation/README.md). The design comes from Kerolos Yacoub's practical session ([Evaluation_practical](https://github.com/Kerolos2019/Evaluation_practical)).

An internal chatbot answers questions about a fictional company's records: CRM exports, projects, revenue and sales-proposal PDFs. The records are synthetic, written for the book by [`make_datasets.py`](make_datasets.py), which rebuilds `datasets/` from scratch. It's **agentic RAG**. Classic RAG retrieves passages before every answer. Here retrieval is a tool, `search_knowledge_base`, that the agent decides to call. For a weather question it calls `get_weather` (live data from [Open-Meteo](https://open-meteo.com), no API key) and searches nothing.

## How it works

**Ingestion** (`src/pipeline.py`, runs once): `datasets/` → `loaders.py` (CSV rows and PDF pages → text) → `chunking.py` (1,000-character chunks, 200 overlap) → `vectorstore.py` (a Chroma collection with OpenAI embeddings; each chunk tagged with its source file).

**Query** (every question): `app.py` (Streamlit chat) → `agents.py` (`create_agent`: a model node and a tool node in a loop) → `tools.py` (`search_knowledge_base` returns the top-4 chunks; `get_weather`) → the answer and the tools used.

```text
datasets/        company files: crm_exports/, projects/, revenue/, proposal_documents/
make_datasets.py rebuilds datasets/ (6 CSV exports, 5 proposal PDFs)
src/
  config.py      paths, model names, chunk size
  loaders.py     CSV / PDF / TXT -> text
  chunking.py    fixed-size chunks with overlap
  vectorstore.py Chroma collection + embeddings
  pipeline.py    ingestion: load -> chunk -> store
  tools.py       search_knowledge_base, get_weather
  agents.py      create_agent; ask_with_tools() -> (answer, tools)
  app.py         Streamlit chat UI
tests/           unit tests (no API keys, no network)
```

## Run it

From the repository root, `pip install -r requirements.txt`. Then copy `.env.example` to `.env` in this folder and fill in `OPENAI_API_KEY`. Add `LANGSMITH_API_KEY` too if you want traces.

From `9-Agentic-RAG/`:

```bash
python -m src.pipeline      # ingest datasets/ into Chroma (once)
streamlit run src/app.py    # chat at http://localhost:8501
```

Try "Which sales rep covers Europe?" (the tools caption shows `search_knowledge_base`) and "What's the weather in London?" (it shows `get_weather`).

Unit tests, from the repository root:

```bash
python -m pytest -q 9-Agentic-RAG/tests
```
