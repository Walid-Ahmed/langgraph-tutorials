# 12. Evaluation — Chapter 14 companion code

Work through these four examples in order:

| Folder | What it teaches | Requirements |
|---|---|---|
| [01-minimal-evaluation](01-minimal-evaluation/README.md) | Two golden cases, one Python target and one custom code evaluator | `langsmith==0.9.8`; no API keys or uploads |
| [02-helpdesk-evaluation](02-helpdesk-evaluation/README.md) | LangGraph help desk: datasets, experiments, component evaluation, regression gates, online feedback and human review | Repository dependencies, OpenAI and LangSmith keys for model/hosted runs |
| [03-agentic-rag](03-agentic-rag/README.md) | The application alone: an agentic RAG chatbot (Chroma + `create_agent`) that decides when to search the knowledge base, with a live weather tool | Repository dependencies and an OpenAI key |
| [04-agentic-rag-evaluation](04-agentic-rag-evaluation/README.md) | The same chatbot, evaluated: golden set, rule-based, heuristic, safety, RAG and LLM-judge metrics, a CI gate, and online scores | Repository dependencies, OpenAI and LangSmith keys |

Start from the repository root:

```bash
pip install langsmith==0.9.8
python 12-Evaluation/01-minimal-evaluation/minimal_eval.py
```

For the help-desk example, install dependencies from the repository root, then switch folders:

```bash
pip install -r requirements.txt
cd 12-Evaluation/02-helpdesk-evaluation
cp .env.example .env
# Fill in your keys before running the hosted examples.
python run_experiment.py v1
python run_experiment.py v2
```

A `.env` in a parent folder also works. Follow the help-desk README for the remaining commands and expected results.

The help-desk unit tests use no model or LangSmith API calls. Run them from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest -q 12-Evaluation/02-helpdesk-evaluation
```

Exercise solutions remain in [`Exercise-Solutions/12-evaluation/`](../Exercise-Solutions/12-evaluation/) and import the help-desk code from its new folder.
