# The smallest evaluation loop: two cases, a target, one evaluator, one evaluate call.
# Why: learn LangSmith's inputs/outputs/score contract before adding a graph or LLM.
# Prerequisite: pip install langsmith==0.9.8
# Run from the repository root: python 12-Evaluation/01-minimal-evaluation/minimal_eval.py
# Expected: exact_match=1 for Stanley Bridge, 0 for the padded Montaza Gate input.
# This uses the real LangSmith SDK locally. No API keys, model calls or uploads.

from uuid import uuid4

from langsmith import Client, evaluate
from langsmith.schemas import Example


def target(inputs: dict) -> dict:
    return {"answer": inputs["text"].lower()}  # deliberately forgets to trim spaces


def exact_match(outputs: dict, reference_outputs: dict) -> dict:
    matches = outputs["answer"] == reference_outputs["answer"]
    return {"score": int(matches), "comment": "matches" if matches else "text differs"}


def make_golden_dataset() -> list[Example]:
    """Return two test inputs with their trusted reference outputs."""
    return [
        Example(
            id=uuid4(),
            inputs={"text": "Stanley Bridge"},
            outputs={"answer": "stanley bridge"},
        ),
        Example(
            id=uuid4(),
            inputs={"text": "  Montaza Gate  "},
            outputs={"answer": "montaza gate"},
        ),
    ]


def main():
    examples = make_golden_dataset()
    results = evaluate(
        target, data=examples, evaluators=[exact_match],
        client=Client(auto_batch_tracing=False),
        upload_results=False,  # local results; the next example stores an experiment
    )
    for row in results:
        result = row["evaluation_results"]["results"][0]
        print(f"{row['example'].inputs['text']!r}: {result.key}={int(result.score)}")


if __name__ == "__main__":
    main()
