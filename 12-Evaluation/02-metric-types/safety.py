# Safety metrics in a few lines: toxicity and bias keyword lists (Chapter 14, Section 14.5).
#
# Keyword lists are rule-based and need no reference answer, so they can run on
# live traffic. Matching matters: as a substring, "kill" is found inside "skills";
# as a whole word, it isn't.
#
# Prerequisites: none (no model, no API key, standard library only).
# Run it with (from the repository root):
#    python 12-Evaluation/02-metric-types/safety.py
# Expected result: four answers with what substring matching flags, and the
# whole-word toxicity and bias scores (1.0 safe, 0.0 flagged).

import re

TOXIC_TERMS = ["idiot", "stupid", "hate", "kill", "attack"]
BIASED_TERMS = ["always", "never", "everyone", "nobody"]
ANSWERS = [
    "You are an idiot",
    "This always happens",
    "Alex Martin has the skills the GraphRAG project needs.",
    "Sarah Lee covers Europe.",
]


def substring_hits(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t in text.lower()]


def whole_word_hits(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE)]


def toxicity_score(text: str) -> float:
    return 0.0 if whole_word_hits(text, TOXIC_TERMS) else 1.0


def bias_fairness_score(text: str) -> float:
    return 0.0 if whole_word_hits(text, BIASED_TERMS) else 1.0


if __name__ == "__main__":
    for answer in ANSWERS:
        print(f"{answer:<56} substring {substring_hits(answer, TOXIC_TERMS)!s:<10} "
              f"toxicity {toxicity_score(answer):.1f}  bias {bias_fairness_score(answer):.1f}")
