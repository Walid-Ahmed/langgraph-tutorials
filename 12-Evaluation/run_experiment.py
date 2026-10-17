# Offline evaluation with LangSmith: the golden set as a dataset, each run an experiment.
#
# This script makes sure the golden set exists in LangSmith as the dataset
# "corniche-cycles-desk" (uploading golden_set.jsonl the first time), then calls
# client.evaluate() to run one version of the help desk over every example with
# three evaluators. It prints one row per example and the averages; the full
# experiment - every trace, score and judge comment - is stored in LangSmith.
#
# Why this matters:
# - This is offline evaluation: a FIXED dataset, the system under test (the
#   "target"), and evaluators. LangSmith runs the loop, stores the results, and
#   links every score to the trace that produced it (Chapter 13).
# - golden_set.jsonl is only the seed. Once uploaded, the LangSmith dataset is the
#   source of truth: new examples are added there (review_queue.py, Section 14.10)
#   and LangSmith keeps a version of the dataset for every change.
# - One run makes about 25 desk calls (gpt-4o-mini) and 12 judge calls (gpt-4o);
#   together they cost a few US cents.
#
# Before you run this:
# 1. pip install -r requirements.txt   (from the repository root)
# 2. A .env file in this folder or the repository root with:
#    OPENAI_API_KEY=your_openai_key
#    LANGSMITH_API_KEY=your_langsmith_key      # from smith.langchain.com
#    LANGSMITH_TRACING=true
#
# Run it with (from the 12-Evaluation folder):
#    python run_experiment.py v1
#    python run_experiment.py v2
#
# Expected result: the first run prints "created dataset corniche-cycles-desk",
# a link to the experiment, a 12-row table and the averages. On our runs:
#   v1: correct_tools 0.83 | facts_present 0.83 | judge_reply 0.83
#   v2: correct_tools 1.00 | facts_present 1.00 | judge_reply 1.00
# The wording of the answers varies between runs; these scores did not.
# After promote_example.py the dataset has 13 examples; v2 still scores 1.00.

import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))  # finds .env here or in a parent folder

from langsmith import Client

from desk import build_desk, run_desk
from evaluators import correct_tools, facts_present, judge_reply

DATASET = "corniche-cycles-desk"
client = Client()


def ensure_dataset() -> None:
    """Upload the seed file the first time; after that the LangSmith copy is the one that counts."""
    if client.has_dataset(dataset_name=DATASET):
        return
    seed = [json.loads(line) for line in Path("golden_set.jsonl").read_text().splitlines()]
    dataset = client.create_dataset(DATASET, description="Corniche Cycles help desk golden set")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[{
            "inputs": {"question": ex["question"]},
            "outputs": {k: ex[k] for k in ("expected_calls", "must_include", "expected_behavior")},
            "metadata": {"id": ex["id"], "tags": ex["tags"]},
        } for ex in seed],
    )
    print(f"created dataset {DATASET} with {len(seed)} examples")


def run(version: str, evaluators=(correct_tools, facts_present, judge_reply)):
    """Run one experiment; return {example id: {evaluator: score}} for printing or gating."""
    desk = build_desk(version)
    results = client.evaluate(
        lambda inputs: run_desk(desk, inputs["question"]),  # the target: inputs -> outputs
        data=DATASET,
        evaluators=list(evaluators),
        experiment_prefix=f"desk-{version}",
        metadata={"version": version},  # lets the UI group experiments, and the gate find them
        max_concurrency=4,
    )
    scores = {}
    for row in results:
        example_id = row["example"].metadata["id"]
        scores[example_id] = {r.key: r.score for r in row["evaluation_results"]["results"]}
    return scores


def print_table(version: str, scores: dict) -> None:
    metrics = ["correct_tools", "facts_present", "judge_reply"]
    print(f"{'example':<16}{'tools':>7}{'facts':>7}{'judge':>7}")
    for example_id, s in sorted(scores.items()):
        print(f"{example_id:<16}" + "".join(f"{s[m]:>7.2f}" for m in metrics))
    averages = {m: sum(s[m] for s in scores.values()) / len(scores) for m in metrics}
    print(f"\n{version}: " + " | ".join(f"{m} {v:.2f}" for m, v in averages.items()))


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v2"
    ensure_dataset()
    print_table(version, run(version))
