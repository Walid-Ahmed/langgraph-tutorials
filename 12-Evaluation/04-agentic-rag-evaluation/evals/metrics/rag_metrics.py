# RAG metrics: did retrieval bring back the right files?
#
# Both compare the sources the agent's searches actually returned with the
# golden case's expected_sources. Precision asks how much of what came back was
# useful; recall asks how much of what was needed came back.

from collections.abc import Sequence

from evals.metrics.result import MetricResult


# Share of retrieved chunks whose source is relevant (repeats count: 4 chunks, 2 useful = 0.5).
def context_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> MetricResult:
    if not retrieved:
        return MetricResult("context_precision", 0.0, "nothing retrieved")
    hits = sum(1 for source in retrieved if source in relevant)
    return MetricResult("context_precision", round(hits / len(retrieved), 3), f"{hits}/{len(retrieved)} chunks relevant")


# Share of the relevant sources that were retrieved at least once.
def context_recall(retrieved: Sequence[str], relevant: Sequence[str]) -> MetricResult:
    if not relevant:
        return MetricResult("context_recall", 1.0, "no sources required")
    missing = [source for source in relevant if source not in retrieved]
    score = (len(relevant) - len(missing)) / len(relevant)
    return MetricResult("context_recall", round(score, 3), f"missing {missing}" if missing else "all sources retrieved")
