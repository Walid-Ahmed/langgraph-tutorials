# Validation shared by the regression gate and its local unit tests.
# Missing examples or scores are an incomplete evaluation, never a passing one.
# Imported by regression_gate.py; no API keys or services are needed.

from math import isfinite
from numbers import Real

DEFAULT_METRICS = ("correct_tools", "facts_present", "judge_reply")


def validate_scores(scores: dict, expected_ids: set, metrics: tuple) -> None:
    """Require one finite score per expected example and selected metric."""
    if not expected_ids or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("expected examples and distinct metrics must be non-empty")
    missing, extra = expected_ids - scores.keys(), scores.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"example coverage mismatch: missing {sorted(missing)}, extra {sorted(extra)}")
    for example_id in sorted(expected_ids):
        for metric in metrics:
            value = scores[example_id].get(metric)
            if not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{example_id}: missing or invalid score for {metric}: {value!r}")
