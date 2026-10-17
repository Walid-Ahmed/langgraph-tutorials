# Regression gate: compare two saved offline runs and fail if anything got worse.
#
# This script reads two result files written by run_offline_eval.py (a baseline
# and a candidate), prints every example whose score changed on any metric, and
# exits with status 1 if the candidate is worse on any example, so a CI job that
# runs it stops the change from merging.
#
# Why this matters:
# - Averages hide regressions. A candidate can fix one example and break
#   another and still tie the baseline's average. Comparing example by example
#   catches the trade.
# - The exit code is the whole CI integration: no plugin, no dashboard. Any CI
#   system treats a non-zero exit as a failed step.
# - A failing gate is a question, not a verdict. Sometimes the golden example is
#   out of date and the right fix is to the dataset, made on purpose and reviewed.
#
# Before you run this:
# 1. Run run_offline_eval.py for both versions first, so results/ has both files.
#
# Run it with (from the 12-Evaluation folder):
#    python compare_runs.py v2 v3
#
# Expected result: one regression, on price-boundary (facts_present and
# judge_reply 1.00 -> 0.00), and "FAIL: v3 is worse than v2 on 1 example(s)".
# `python compare_runs.py v1 v2` prints two improvements and "PASS".

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load(version: str) -> dict:
    data = json.loads((HERE / "results" / f"{version}.json").read_text())
    return {row["id"]: row["scores"] for row in data["rows"]}


def compare(baseline: str, candidate: str) -> int:
    old, new = load(baseline), load(candidate)
    worse = set()
    for example_id, old_scores in old.items():
        for metric, old_score in old_scores.items():
            before = old_score["score"]
            after = new[example_id][metric]["score"]
            if after == before:
                continue
            tag = "REGRESSED" if after < before else "improved "
            print(f"{tag} {example_id:<16}{metric:<15}{before:.2f} -> {after:.2f}")
            if after < before:
                worse.add(example_id)
                print(f"          reason: {new[example_id][metric]['reason']}")
    if worse:
        print(f"\nFAIL: {candidate} is worse than {baseline} on {len(worse)} example(s)")
        return 1
    print(f"\nPASS: {candidate} is no worse than {baseline} on any example")
    return 0


if __name__ == "__main__":
    sys.exit(compare(sys.argv[1], sys.argv[2]))
