# Helpers shared by the Chapter 14 solutions: work on a COPY of the golden dataset.
#
# What it does:
#   ensure_copy() makes a LangSmith dataset that starts as a copy of the
#   chapter's "corniche-cycles-desk" dataset, then adds any extra golden
#   examples (in golden_set.jsonl's shape) that the copy doesn't have yet.
#   use_dataset() points the chapter's run_experiment.run() and
#   regression_gate.gate() at that copy.
#
# Why it matters:
#   - The exercises add examples and run experiments. Doing that on the chapter's
#     own dataset would change the numbers the chapter reports and the baseline
#     that regression_gate.py compares against. A copy keeps both untouched.
#   - Running ensure_copy() twice does nothing the second time. Examples are
#     matched by their metadata "id", so a re-run never adds duplicates.
#   - use_dataset() reuses the chapter's harness instead of rewriting it. Both
#     run() and gate() read the module-level DATASET when they are called.
#
# Prerequisites:
#   LANGSMITH_API_KEY in the environment. This module is imported by the ex*.py
#   scripts in this folder and is not run on its own.

import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "12-Evaluation" / "02-helpdesk-evaluation"
sys.path.insert(0, str(EVAL_DIR))  # the chapter's desk, evaluators and harness

SOURCE = "corniche-cycles-desk"


def to_example(golden: dict) -> dict:
    """One golden_set.jsonl line in the shape LangSmith stores (as run_experiment.ensure_dataset does)."""
    return {
        "inputs": {"question": golden["question"]},
        "outputs": {k: golden[k] for k in ("expected_calls", "must_include", "expected_behavior")},
        "metadata": {"id": golden["id"], "tags": golden["tags"]},
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ensure_copy(client, target: str, extra: list[dict] | None = None) -> int:
    """Create `target` as a copy of the chapter's dataset if needed, add missing extras; return its size."""
    if not client.has_dataset(dataset_name=target):
        dataset = client.create_dataset(target, description=f"Exercise copy of {SOURCE}")
        client.create_examples(dataset_id=dataset.id, examples=[
            {"inputs": ex.inputs, "outputs": ex.outputs, "metadata": ex.metadata}
            for ex in client.list_examples(dataset_name=SOURCE)
        ])
        print(f"created {target} as a copy of {SOURCE}")
    have = {ex.metadata["id"] for ex in client.list_examples(dataset_name=target)}
    new = [to_example(g) for g in extra or [] if g["id"] not in have]
    if new:
        dataset_id = client.read_dataset(dataset_name=target).id
        client.create_examples(dataset_id=dataset_id, examples=new)
        print(f"added {[ex['metadata']['id'] for ex in new]} to {target}")
    return len(have) + len(new)


def wait_for_experiment(client, dataset: str, version: str, metrics: list[str], timeout=180):
    """Wait until the latest desk-<version> experiment's runs and feedback can be read back.

    client.evaluate() returns as soon as the scores are computed, but LangSmith
    ingests runs and feedback in the background, so reading the experiment back
    straight away (as regression_gate.baseline_scores does) can find nothing yet.
    Returns the experiment's name and its root runs.
    """
    size = sum(1 for _ in client.list_examples(dataset_name=dataset))
    deadline = time.time() + timeout
    while True:
        experiments = [p for p in client.list_projects(reference_dataset_name=dataset,
                                                       metadata={"version": version})
                       if p.name.startswith(f"desk-{version}-")]
        latest = max(experiments, key=lambda p: p.start_time)
        runs = list(client.list_runs(project_name=latest.name, is_root=True))
        if len(runs) == size and all(set(metrics) <= set(r.feedback_stats or {}) for r in runs):
            return latest.name, runs
        if time.time() > deadline:
            sys.exit(f"{latest.name}: runs or feedback still missing after {timeout}s")
        time.sleep(5)


def show_replies(client, dataset: str, version: str, ids: set[str], metrics: list[str]) -> None:
    """Print route, tool results, reply and evaluator comments for some examples of the latest experiment."""
    name, all_runs = wait_for_experiment(client, dataset, version, metrics)
    names = {ex.id: ex.metadata["id"] for ex in client.list_examples(dataset_name=dataset)}
    runs = [r for r in all_runs if names[r.reference_example_id] in ids]
    comments = {}
    for fb in client.list_feedback(run_ids=[r.id for r in runs]):
        comments.setdefault(fb.run_id, []).append(f"{fb.key}={fb.score:g}: {fb.comment}")
    for r in sorted(runs, key=lambda r: names[r.reference_example_id]):
        print(f"\n--- {names[r.reference_example_id]}  ({name})")
        print(f"tools:        {r.outputs['tools_called']}")
        print(f"tool results: {json.dumps(r.outputs['tool_results'])}")
        print(f"reply:        {r.outputs['answer']}")
        for line in sorted(comments.get(r.id, [])):
            print(f"  {line}")


def use_dataset(name: str) -> None:
    """Point the chapter's run() and gate() at another dataset."""
    import regression_gate
    import run_experiment
    run_experiment.DATASET = name
    regression_gate.DATASET = name
