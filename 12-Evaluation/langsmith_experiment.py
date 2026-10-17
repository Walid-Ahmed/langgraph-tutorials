# The same offline evaluation, run as a LangSmith experiment.
#
# This script uploads golden_set.jsonl to LangSmith as a dataset (once), then
# calls client.evaluate() to run one version of the help desk over it with the
# same three evaluators. Every example's trace, scores and judge reasons land in
# a named experiment you can open, filter and compare side by side in the UI.
#
# Why this matters:
# - Nothing about WHAT is measured changes from run_offline_eval.py. LangSmith
#   adds storage, a UI and side-by-side comparison of experiments; the dataset
#   and the evaluators are still yours.
# - The evaluators are thin wrappers: LangSmith passes `inputs`, `outputs` and
#   `reference_outputs` by name, and our functions already take those three
#   things under slightly different names.
# - Each example's run is a full trace (Chapter 13), so a low score is one click
#   away from the model call that caused it.
#
# Before you run this:
# 1. pip install -r requirements.txt   (from the repository root)
# 2. A .env file in this folder or the repository root with:
#    OPENAI_API_KEY=your_openai_key
#    LANGSMITH_API_KEY=your_langsmith_key      # from smith.langchain.com
#    LANGSMITH_TRACING=true
#
# Run it with (from the 12-Evaluation folder):
#    python langsmith_experiment.py v2
#    python langsmith_experiment.py v3
#
# Expected result: the first run prints "created dataset corniche-cycles-golden";
# every run prints a link to the experiment (desk-v2-xxxx) and the same averages
# as run_offline_eval.py for that version.

import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from langsmith import Client

from desk import build_desk, run_desk
from evaluators import correct_tools, facts_present, judge_reply

DATASET = "corniche-cycles-golden"
client = Client()


def ensure_dataset() -> None:
    """Upload the golden set the first time; later runs reuse the stored copy."""
    if client.has_dataset(dataset_name=DATASET):
        return
    golden = [json.loads(line) for line in Path("golden_set.jsonl").read_text().splitlines()]
    dataset = client.create_dataset(DATASET, description="Corniche Cycles help desk golden set")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[{
            "inputs": {"question": ex["question"]},
            "outputs": {k: ex[k] for k in ("expected_tools", "must_include", "expected_behavior")},
            "metadata": {"id": ex["id"], "tags": ex["tags"]},
        } for ex in golden],
    )
    print(f"created dataset {DATASET} with {len(golden)} examples")


# LangSmith matches evaluator arguments by NAME: inputs, outputs, reference_outputs.
def route(outputs, reference_outputs):
    result = correct_tools(outputs, reference_outputs)
    return {"score": result["score"], "comment": result["reason"]}


def facts(outputs, reference_outputs):
    result = facts_present(outputs, reference_outputs)
    return {"score": result["score"], "comment": result["reason"]}


def judge(inputs, outputs, reference_outputs):
    result = judge_reply(inputs, outputs, reference_outputs)
    return {"score": result["score"], "comment": result["reason"]}


def main(version: str):
    ensure_dataset()
    desk = build_desk(version)
    results = client.evaluate(
        lambda inputs: run_desk(desk, inputs["question"]),  # the target: inputs -> outputs
        data=DATASET,
        evaluators=[route, facts, judge],
        experiment_prefix=f"desk-{version}",
        metadata={"version": version},
        max_concurrency=4,
    )
    scores: dict[str, list[float]] = {}
    for row in results:
        for res in row["evaluation_results"]["results"]:
            scores.setdefault(res.key, []).append(res.score)
    print(f"{version}: " + " | ".join(f"{k} {sum(v) / len(v):.2f}" for k, v in scores.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v2")
