# A minimal LangSmith evaluation

An optional warm-up for Chapter 14, before the [agentic RAG evaluation](../04-agentic-rag-evaluation/README.md). This example tests an ordinary Python function; it needs no LLM, graph, API keys or uploads.

From the repository root:

```bash
pip install langsmith==0.9.8
python 12-Evaluation/01-minimal-evaluation/minimal_eval.py
```

The two score lines are:

```text
'Stanley Bridge': exact_match=1
'  Montaza Gate  ': exact_match=0
```

SDK progress messages and a beta warning for `upload_results` may also appear. Results may be printed in a different order.

## What the code evaluates

`target` should lowercase a station name and remove surrounding spaces. It deliberately uses only `.lower()`. The first input has no padding and passes; the second retains its spaces and fails. Change the target expression to `inputs["text"].strip().lower()` and both scores become 1.

`make_golden_dataset()` is a separate function that returns two in-memory `Example` objects. Each has test `inputs`, trusted reference `outputs`, and an identifier. These input/reference pairs are the golden dataset. They do not appear in the LangSmith Datasets UI.

`exact_match` is our own evaluator, defined in this file above `main()`. It compares the actual answer with the reference answer and returns a numeric score and a comment. No judge model decides these scores.

## What we pass to evaluate

| Argument | Value | Role |
|---|---|---|
| `target` | The function reference `target` | Produces the actual answer |
| `data` | `examples` | Supplies the test inputs and reference outputs |
| `evaluators` | `[exact_match]` | Supplies our grading function |
| `client` | `Client(auto_batch_tracing=False)` | Avoids starting a background upload worker |
| `upload_results` | `False` | Keeps evaluation results local |

We pass function references, without calling them. LangSmith calls `target` with each example's inputs, calls `exact_match` with the returned outputs and the matching reference outputs, and collects the scores/comments. The target does not receive the reference answer. Because our evaluator returns no feedback key, the SDK uses its function name, `exact_match`.

The loop in `main()` reads the first evaluator result from each row and prints its key and score. A score of 0 is data: this printing-only script still exits normally with status 0. An explicit score check, like the gate in `04-agentic-rag-evaluation/evals/gate.py`, is needed to fail a test or CI step.

LangSmith supplies the evaluation loop. We supply the cases, the application logic and the definition of correctness. With `upload_results=False`, this example creates no hosted dataset or stored experiment. The next example uploads results so you can inspect them in the UI.
