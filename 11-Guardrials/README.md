# 11. Guardrails — Checks Around an Agent with LangChain Middleware

**Book mapping:** Chapter 15 — Guardrails. Every listing in the chapter (15.1–15.9) is trimmed from one notebook: [`guardrails_langchain_middleware.ipynb`](guardrails_langchain_middleware.ipynb).

**Requires:** `OPENAI_API_KEY` in the repo-root `.env`, and Jupyter (not in `requirements.txt`): `pip install notebook`, or open the notebook in VS Code with the repo's `.venv` as the kernel.

Tutorial 10 let you *see* what an agent did. This one limits what it is *allowed* to do: checks that run before the model, after the model, and before a tool acts, all attached to a `create_agent` agent as middleware.

## What's in the notebook

The notebook runs top to bottom. Its sections line up with the chapter like this:

| Notebook section | What it shows | Book |
|---|---|---|
| Deterministic vs model-based | The same three inputs checked by a banned-word list and by a model | §15.2, Listings 15.1–15.2 |
| PII detection middleware | Redact, mask and block emails, cards and a custom pattern | §15.4, Listings 15.3–15.4 |
| Human-in-the-loop middleware | Pause before risky tools; approve or reject the call | §15.7, Listings 15.7–15.8 |
| Before-agent guardrail | A cheap input filter that ends the run before the first model call | §15.5, Listing 15.5 |
| After-agent guardrail | A model grades the answer against one stated rule | §15.6, Listing 15.6 |
| Layered guardrails: healthcare chatbot | Four guardrails on one assistant, cheapest first | §15.8, Listing 15.9 |

## Running it

From the repo root, with the virtual environment active:

```bash
pip install notebook
jupyter notebook 11-Guardrials/guardrails_langchain_middleware.ipynb
```

Run the cells in order. The human-in-the-loop cells pause the agent on purpose; the next cell resumes it with `Command(resume=...)` on the same `thread_id`.

To watch the guarded runs in LangSmith (Exercise 15.4), add `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` to the same `.env` before you start Jupyter. Tutorial 10 explains the three variables.

## Next step

[Tutorial 12 — Evaluation](../12-Evaluation/README.md): measure how well the agent answers, with a golden set and metrics, before and after every change.
