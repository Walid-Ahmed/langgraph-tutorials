# Heuristic metrics: cheap approximations based on word overlap and length.
# They need no model call, but they reward wording, not meaning: a correct answer
# phrased differently from the reference scores low.

import re

from evals.metrics.result import MetricResult


# The distinct words of a text, lowercased, without punctuation.
def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


# ROUGE-1-style recall: the share of the reference's distinct words the answer contains.
def rouge(candidate: str, reference: str) -> MetricResult:
    reference_words = _words(reference)
    shared = _words(candidate) & reference_words
    score = len(shared) / max(len(reference_words), 1)
    return MetricResult("rouge", round(score, 3), f"{len(shared)}/{len(reference_words)} reference words")


# 1.0 if the answer has at least min_words words, else 0.0.
def length_penalty(text: str, min_words: int = 20) -> MetricResult:
    count = len(text.split())
    return MetricResult("length_penalty", 1.0 if count >= min_words else 0.0, f"{count} words")
