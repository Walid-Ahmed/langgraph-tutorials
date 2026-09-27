# 03. Multi-tool agent — production-shaped package

This is file 03 of [tutorial 10](../README.md). LangSmith tracing is still the lesson; the layout is how a small production agent is usually split.

```text
03_multi_tool_agent/
  research_assistant.py  # SYSTEM_PROMPT + build_research_assistant()
  tools.py               # search_local_docs, google_search
  local_RAG/             # .txt files indexed for search_local_docs (employees/)
  diagrams/              # PNG from plot_graph (written when you run main.py)
  .env                   # optional: LANGSMITH_PROJECT for this package only
  main.py                # demo: QUERIES, FAISS, graph PNG, invoke
```

| File | Owns |
|---|---|
| `research_assistant.py` | `SYSTEM_PROMPT` and `build_research_assistant()` |
| `tools.py` | what the model may call |
| `main.py` | three hardcoded questions + `run_name` |

`create_agent` builds the model ↔ tools loop. You do not add a `ToolNode` or router here. A server would call `build_research_assistant(vectorstore)` the same way `main.py` does, then `invoke` once per user request. Input state is explicit in `main.py`: `MessagesState` with key `messages` (the same shape file 01 would call `State`, but here the field name is fixed by `create_agent`).

The numbered folder name is for tutorial order. A real package would not start with `03_` (Python cannot `import 03_multi_tool_agent`).

## Prerequisites

Local RAG as a **workflow** first: [`../../5-Workflows/01_rag_retrieve_generate.py`](../../5-Workflows/01_rag_retrieve_generate.py). Same index helper: [`../../rag_index.py`](../../rag_index.py) `build_vectorstore` over [`local_RAG/`](local_RAG/) (`**/*.txt`, including `employees/`). Keys in [`../.env`](../.env): `OPENAI_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, optional `SERPER_API_KEY`. Override the LangSmith **project name** in this folder's [`.env`](.env) (copy [`.env.example`](.env.example)): `LANGSMITH_PROJECT=...`. `main.py` loads the parent file first, then this one (`override=True`).

There is no `@traceable` in this package. `create_agent`, `ChatOpenAI`, and the tools are recorded automatically once those env vars are set. `main.py` loads `.env` **before** importing LangChain, then names each invoke with `config={"run_name": "..."}`.

## Run

From `10-observability/`:

```bash
python 03_multi_tool_agent/main.py
```

`main.py` also calls [`plot_graph`](../../util.py) and writes [`diagrams/03_multi_tool_agent.png`](diagrams/03_multi_tool_agent.png). The terminal prints `Graph saved to` and that path.

Three traces under whatever `LANGSMITH_PROJECT` is in this folder's `.env` (default `10-observability`): **Multi-Tool: local docs**, **Multi-Tool: web search**, and **Multi-Tool: both**. Open them side by side and check which tools ran.
