# Types of metrics, in a few lines each

Companion code for **Chapter 14**, Section 14.6 of *Building AI Agents with LangGraph*. Each script shows one family of evaluation metrics on small, fixed examples, before the chapter applies them to the RAG chatbot in [`04-agentic-rag-evaluation`](../04-agentic-rag-evaluation/README.md). Several examples come from Kerolos Yacoub's practical evaluation session.

| Script | Family | Shows | Needs |
|---|---|---|---|
| [`rule_and_heuristic.py`](rule_and_heuristic.py) | Rule-based and heuristic | tool choice, keyword coverage, exact match; word precision (BLEU-1 style), word recall (ROUGE-1 style), a length check | Nothing |
| [`safety.py`](safety.py) | Safety | toxicity and bias keyword lists; substring vs whole-word matching | Nothing |
| [`rag.py`](rag.py) | RAG (retrieval) | precision@k, recall@k, reciprocal rank | Nothing |
| [`llm_judge.py`](llm_judge.py) | LLM-as-judge | a rubric, structured output, the reason before the score | `OPENAI_API_KEY` |

Run any of them from the repository root:

```bash
python 12-Evaluation/02-metric-types/rule_and_heuristic.py
python 12-Evaluation/02-metric-types/safety.py
python 12-Evaluation/02-metric-types/rag.py
python 12-Evaluation/02-metric-types/llm_judge.py
```

The point of the first script's table: word recall gives a right answer and a wrong answer the same score, and only the keyword check (and, in `llm_judge.py`, the judge) tells them apart.
