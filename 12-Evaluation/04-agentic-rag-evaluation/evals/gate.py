# The CI gate's rules, kept apart from the agent so they can be unit-tested.
#
# A case passes when the judge rates it correct (>= 0.8) AND the agent called the
# expected tool first. The run passes when at least 80% of cases pass AND no case
# that passed in the reviewed baseline fails now: a pass rate can hold steady while
# one case breaks and another gets fixed, so the gate also compares case by case.
# Imported by run_eval.py, not run.

PASS_THRESHOLD = 0.8   # minimum correctness score for a case to pass
MIN_PASS_RATE = 0.8    # minimum share of passing cases for the run to pass


# A case passes when the judge rates it correct and the right tool came first.
def case_passed(scores: dict) -> bool:
    return scores["correctness"] >= PASS_THRESHOLD and scores["tool_call_accuracy"] == 1.0


# The pass count, the pass rate and the average of every metric over the run.
def summarize(cases: list[dict]) -> dict:
    metric_names = list(cases[0]["scores"])
    passed = sum(1 for case in cases if case["passed"])
    return {
        "total_cases": len(cases),
        "passed_cases": passed,
        "pass_rate": round(passed / len(cases), 3),
        "average_scores": {name: round(sum(c["scores"][name] for c in cases) / len(cases), 3)
                           for name in metric_names},
    }


# Ids of cases that passed in the baseline and fail (or are missing) now.
def find_regressions(baseline_cases: list[dict], current_cases: list[dict]) -> list[str]:
    current = {case["id"]: case["passed"] for case in current_cases}
    return [case["id"] for case in baseline_cases if case["passed"] and not current.get(case["id"], False)]
