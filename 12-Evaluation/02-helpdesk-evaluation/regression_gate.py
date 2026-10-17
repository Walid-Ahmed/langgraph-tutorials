# Compare a candidate with a stored baseline on the SAME dataset snapshot.
#
# Why this matters:
# - Compare each example, not averages. Missing runs/feedback must fail, not pass.
# - The baseline's dataset version pins the candidate's inputs and references.
# - Select a reviewed experiment explicitly for CI; the version-only shorthand
#   chooses the latest matching experiment for interactive use.
# - A nonzero exit fails the CI step; require that check in repository rules
#   if a failure must prevent merging.
#
# Prerequisites: pip install -r requirements.txt (repository root), the chapter's
# .env keys, and a stored v2 baseline (`python run_experiment.py v2`).
# Run from 12-Evaluation/02-helpdesk-evaluation:
#   python regression_gate.py v2 v3
#   python regression_gate.py v2 v3 --baseline-experiment desk-v2-<suffix>
# Expected: price-boundary regresses, FAIL, exit 1. Incomplete results also fail.

import argparse
import sys

from langsmith import Client

from gate_checks import DEFAULT_METRICS, validate_scores
from run_experiment import DATASET, run

client = Client()


def baseline_scores(version: str, experiment: str | None = None, metrics=DEFAULT_METRICS):
    """Read a complete baseline and its historical dataset version."""
    if experiment:
        latest = client.read_project(project_name=experiment)
        dataset = client.read_dataset(dataset_name=DATASET)
        if latest.reference_dataset_id != dataset.id or latest.metadata.get("version") != version:
            raise ValueError("baseline experiment must match the dataset and version")
    else:
        experiments = [p for p in client.list_projects(reference_dataset_name=DATASET,
                                                       metadata={"version": version})
                       if p.name.startswith(f"desk-{version}-")]
        if not experiments:
            raise ValueError(f"no stored baseline for {version}; run its experiment first")
        latest = max(experiments, key=lambda p: p.start_time)
    snapshot = latest.metadata.get("dataset_version")
    if not snapshot:
        raise ValueError("baseline has no dataset_version; wait for ingestion or rerun it")
    examples = list(client.list_examples(dataset_name=DATASET, as_of=snapshot))
    names = {ex.id: (ex.metadata or {}).get("id") for ex in examples}
    if any(not isinstance(name, str) or not name for name in names.values()):
        raise ValueError("every baseline example needs a non-empty metadata id")
    if len(set(names.values())) != len(names):
        raise ValueError("baseline example metadata ids must be unique")
    scores = {}
    for result in client.list_runs(project_name=latest.name, is_root=True):
        if result.reference_example_id not in names or result.error:
            raise ValueError("baseline contains an unknown example or a failed run")
        example_id = names[result.reference_example_id]
        if example_id in scores:
            raise ValueError("baseline has repeated runs; this gate expects one run per example")
        scores[example_id] = {key: stats.get("avg")
                              for key, stats in (result.feedback_stats or {}).items()}
    validate_scores(scores, set(names.values()), metrics)
    return latest.name, snapshot, scores


def gate(baseline: str, candidate: str, *, experiment=None,
         metrics=DEFAULT_METRICS, evaluators=None) -> int:
    try:
        baseline_name, snapshot, old = baseline_scores(baseline, experiment, metrics)
        print(f"baseline: {baseline_name}")
        new = run(candidate, evaluators=evaluators, dataset_version=snapshot)
        validate_scores(new, set(old), metrics)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    worse = set()
    for example_id in sorted(old):
        for metric in sorted(metrics):
            before, after = old[example_id][metric], new[example_id][metric]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--baseline-experiment", help="the exact reviewed experiment name")
    args = parser.parse_args()
    sys.exit(gate(args.baseline, args.candidate, experiment=args.baseline_experiment))
