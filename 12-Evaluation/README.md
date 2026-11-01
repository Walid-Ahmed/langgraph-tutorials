# 12. Evaluation — Chapter 14 companion code

| Folder | What it teaches | Requirements |
|---|---|---|
| [03-agentic-rag](03-agentic-rag/README.md) | The application alone: an agentic RAG chatbot (Chroma + `create_agent`) that decides when to search the knowledge base, with a live weather tool | Repository dependencies and an OpenAI key |
| [04-agentic-rag-evaluation](04-agentic-rag-evaluation/README.md) | The same chatbot, evaluated: golden set, rule-based, heuristic, safety, RAG and LLM-judge metrics, a CI gate, online scores and human review | Repository dependencies, OpenAI and LangSmith keys |
| [01-minimal-evaluation](01-minimal-evaluation/README.md) | Optional warm-up: two golden cases, one Python target and one custom code evaluator | `langsmith==0.9.8`; no API keys or uploads |

Install the dependencies from the repository root, then follow each folder's README:

```bash
pip install -r requirements.txt
```

The unit tests use no model, LangSmith or network calls. Run them from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest -q 12-Evaluation/03-agentic-rag/tests
python -m pytest -q 12-Evaluation/04-agentic-rag-evaluation/tests
```

Exercise solutions are in [`Exercise-Solutions/12-evaluation-agentic-rag/`](../Exercise-Solutions/12-evaluation-agentic-rag/).
