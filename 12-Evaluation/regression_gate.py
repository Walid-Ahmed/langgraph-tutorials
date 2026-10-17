# Regression gate: run a candidate experiment and fail if any example got worse
# than in the baseline experiment already stored in LangSmith.
#
# This script finds the most recent "desk-<baseline>" experiment on the
# corniche-cycles-desk dataset, reads its per-example scores back from LangSmith,
# runs a new experiment for the candidate version, compares the two example by
# example, and exits with status 1 if the candidate is worse anywhere, so a CI
# job that runs it stops the change from merging.
#
# Why this matters:
# - Averages hide regressions. A candidate can fix one example and break another
#   and still tie the baseline's average. Comparing example by example catches it.
# - The baseline is not re-run: it's the experiment you already reviewed, stored
#   with its scores, its traces, the dataset version and the git commit it ran on.
# - The exit code is the whole CI integration. Any CI system treats a non-zero
#   exit as a failed step.
# - A failing gate is a question, not a verdict. Sometimes the golden example is
#   out of date and the right fix is to the dataset, made on purpose and reviewed.
#
# Before you run this:
# 1. Run `python run_experiment.py v2` first, so a v2 baseline experiment exists.
# 2. The same .env as run_experiment.py.
#
# Run it with (from the 12-Evaluation folder):
#    python regression_gate.py v2 v3
#
# Expected result: the baseline experiment's name, a link to the new v3
# experiment, one regression on price-boundary (facts_present and judge_reply
# 1.00 -> 0.00) and "FAIL: v3 is worse than v2 on 1 example(s)". Exit status 1.

import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from langsmith import Client

from run_experiment import DATASET, run

client = Client()


def baseline_scores(version: str) -> tuple[str, dict]:
    """Per-example scores of the latest stored experiment for `version`."""
    experiments = [p for p in client.list_projects(reference_dataset_name=DATASET,
                                                   metadata={"version": version})
                   if p.name.startswith(f"desk-{version}-")]
    latest = max(experiments, key=lambda p: p.start_time)
    names = {ex.id: ex.metadata["id"] for ex in client.list_examples(dataset_name=DATASET)}
    scores = {}
    for r in client.list_runs(project_name=latest.name, is_root=True):
        stats = r.feedback_stats or {}
        scores[names[r.reference_example_id]] = {key: s["avg"] for key, s in stats.items()}
    return latest.name, scores


def gate(baseline: str, candidate: str) -> int:
    baseline_name, old = baseline_scores(baseline)
    print(f"baseline: {baseline_name}")
    new = run(candidate)
    worse = set()
    for example_id, old_scores in sorted(old.items()):
        for metric in sorted(old_scores.keys() & new.get(example_id, {}).keys()):
            before, after = old_scores[metric], new[example_id][metric]
            if after < before:
                worse.add(example_id)
                print(f"REGRESSED {example_id:<16}{metric:<15}{before:.2f} -> {after:.2f}")
            elif after > before:
                print(f"improved  {example_id:<16}{metric:<15}{before:.2f} -> {after:.2f}")
    if worse:
        print(f"\nFAIL: {candidate} is worse than {baseline} on {len(worse)} example(s)")
        return 1
    print(f"\nPASS: {candidate} is no worse than {baseline} on any example")
    return 0


if __name__ == "__main__":
    sys.exit(gate(sys.argv[1], sys.argv[2]))
