# Rule-based metrics: exact, deterministic checks. Free, instant, same score every run.

from collections.abc import Sequence

from evals.metrics.result import MetricResult


def tool_call_accuracy(actual_tool: str | None, expected_tool: str | None) -> MetricResult:
    """1.0 if the first tool the agent called is the expected one, else 0.0."""
    hit = actual_tool is not None and actual_tool == expected_tool
    return MetricResult("tool_call_accuracy", 1.0 if hit else 0.0,
                        f"called {actual_tool or 'no tool'}, expected {expected_tool or 'no tool'}")


def keyword_coverage(answer: str, expected_keywords: Sequence[str]) -> MetricResult:
    """Share of the expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return MetricResult("keyword_coverage", 1.0, "no keywords required")
    answer_lower = answer.lower()
    missing = [k for k in expected_keywords if k.lower() not in answer_lower]
    score = (len(expected_keywords) - len(missing)) / len(expected_keywords)
    return MetricResult("keyword_coverage", round(score, 3),
                        f"missing {missing}" if missing else "all keywords present")
