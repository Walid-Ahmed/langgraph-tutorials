# Heuristic metrics: cheap approximations based on word overlap and length.
# They need no model call, but they reward wording, not meaning: a correct answer
# phrased differently from the reference scores low.

import re

from evals.metrics.result import MetricResult


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def rouge(candidate: str, reference: str) -> MetricResult:
    """ROUGE-1-style recall: the share of the reference's distinct words the answer contains."""
    reference_words = _words(reference)
    shared = _words(candidate) & reference_words
    score = len(shared) / max(len(reference_words), 1)
    return MetricResult("rouge", round(score, 3), f"{len(shared)}/{len(reference_words)} reference words")


def length_penalty(text: str, min_words: int = 20) -> MetricResult:
    """1.0 if the answer has at least min_words words, else 0.0."""
    count = len(text.split())
    return MetricResult("length_penalty", 1.0 if count >= min_words else 0.0, f"{count} words")
