# 01. Local RAG — Retrieve Then Generate

This example is **prompt chaining** where the first step is retrieval, not an LLM call. You decide the path at build time: every question hits the local index, then the model answers from those chunks.

This is **not** an agent. The model never chooses whether to search. The same index is a tool in [`../6-Agents/04_rag_as_tool.py`](../6-Agents/04_rag_as_tool.py) and in [`../10-observability/03_multi_tool_agent/`](../10-observability/03_multi_tool_agent/) so the model decides.

## Part 1 — Core Tutorial

Local RAG on disk usually looks like this:

```text
folder of .txt files → split → embed → FAISS index
                              (once, at startup)

question → retrieve top-k chunks → generate answer from chunks
                              (every invoke)
```

`build_vectorstore(docs_dir)` in [`../rag_index.py`](../rag_index.py) does the first line. The LangGraph does the second:

```mermaid
flowchart LR
    START([START]) --> R["retrieve"]
    R --> G["generate"]
    G --> END([END])
```

| Stage | Reads | Writes |
|---|---|---|
| `retrieve` | `question` | `retrieved` (top 3 chunks) |
| `generate` | `question`, `retrieved` | `answer` |

The index is built **once** in `main()`, then closed over by the retrieve node. Do not rebuild FAISS inside the node — that would re-embed on every question.

## When To Use

Use this workflow when **every** question should consult the same private docs: a product guide, policy handbook, or internal wiki.

Use an agent with a retrieve **tool** ([`04_rag_as_tool.py`](../6-Agents/04_rag_as_tool.py), [`03_multi_tool_agent/`](../10-observability/03_multi_tool_agent/)) when the model might need docs, the web, or no tool at all.

## What To Look For In The Code Example

| Concept | Code Name |
|---|---|
| Folder → FAISS | `build_vectorstore(docs_dir)` |
| State | `RAGState` (`question`, `retrieved`, `answer`) |
| Always-on retrieval | `retrieve` node |
| Grounded answer | `generate` node |
| Wiring | `START → retrieve → generate → END` |

Run from the repo root:

```bash
python "5-Workflows/01_rag_retrieve_generate.py"
```

Needs `OPENAI_API_KEY` (chat + embeddings). Default docs folder: [`data/`](data/) (`llm_production_guide.txt`). Pass any other folder of `.txt` files to `build_vectorstore`.
