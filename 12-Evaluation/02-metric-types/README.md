# Types of metrics, in a few lines each

Companion code for **Chapter 14**, Section 14.6 of *Building AI Agents with LangGraph*. Each script shows one family of evaluation metrics on small, fixed examples from an imaginary travel assistant, kept apart from the RAG chatbot the chapter evaluates later in [`04-agentic-rag-evaluation`](../04-agentic-rag-evaluation/README.md). The safety examples and the doc1-doc5 retrieval example come from Kerolos Yacoub's practical evaluation session.

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

The point of the first script's table: both word-overlap heuristics score a wrong answer higher than a right one, and only the keyword check (and, in `llm_judge.py`, the judge) picks the right one.
