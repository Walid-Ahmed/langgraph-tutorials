# 12. Evaluation — Chapter 14 companion code

| Folder | What it teaches | Requirements |
|---|---|---|
| [02-metric-types](02-metric-types/README.md) | One small script per metric family: rule-based, heuristic, safety, RAG and LLM-as-judge (book Section 14.6) | Nothing, except an OpenAI key for `llm_judge.py` |
| [03-first-langsmith-eval](03-first-langsmith-eval/README.md) | The whole loop at its smallest: five golden cases, one scorer, its wrapper and one LangSmith experiment (book Section 14.7) | OpenAI and LangSmith keys |
| [04-agentic-rag-evaluation](04-agentic-rag-evaluation/README.md) | The Chapter 12 chatbot ([`9-Agentic-RAG`](../9-Agentic-RAG/README.md)), evaluated: golden set, rule-based, heuristic, safety, RAG and LLM-judge metrics, a CI gate, online scores and human review | Repository dependencies, OpenAI and LangSmith keys |
| [01-minimal-evaluation](01-minimal-evaluation/README.md) | Optional warm-up: two golden cases, one Python target and one custom code evaluator | `langsmith==0.9.8`; no API keys or uploads |

Install the dependencies from the repository root, then follow each folder's README:

```bash
pip install -r requirements.txt
```

The unit tests use no model, LangSmith or network calls. Run them from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest -q 12-Evaluation/04-agentic-rag-evaluation/tests
```

Exercise solutions are in [`Exercise-Solutions/12-evaluation-agentic-rag/`](../Exercise-Solutions/12-evaluation-agentic-rag/).
