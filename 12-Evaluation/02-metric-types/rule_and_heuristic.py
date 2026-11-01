# Rule-based and heuristic metrics in a few lines of plain Python (Chapter 14, Section 14.5).
#
# Rule-based metrics check a condition the answer must meet: the right tool, the
# required keywords, an exact match. Heuristic metrics measure word overlap and
# length as a stand-in for quality. Two answers to "Which sales rep covers Europe?"
# are scored against "Sarah Lee covers Europe.": one right but reworded, one wrong
# but worded like the reference. The output shows why heuristics shouldn't decide
# on their own whether an answer passes.
#
# Prerequisites: none (no model, no API key, standard library only).
# Run it with (from the repository root):
#    python 12-Evaluation/02-metric-types/rule_and_heuristic.py
# Expected result: three rule-based checks, then a table of five scores per answer;
# word recall (ROUGE-1 style) gives both answers 0.75.

import re

REFERENCE = "Sarah Lee covers Europe."
KEYWORDS = ["Sarah Lee", "Europe"]
ANSWERS = {
    "A (right)": "Europe is handled by Sarah Lee.",
    "B (wrong)": "Sarah Lee covers North America.",
}


# Rule-based: the answer meets a condition, or it doesn't.
def tool_call_accuracy(actual_tool: str | None, expected_tool: str) -> float:
    return 1.0 if actual_tool == expected_tool else 0.0


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    found = [k for k in keywords if k.lower() in answer.lower()]
    return len(found) / len(keywords)


def exact_match(answer: str, reference: str) -> float:
    return 1.0 if answer.strip() == reference.strip() else 0.0


# Heuristic: surface statistics that stand in for quality.
def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def word_precision(answer: str, reference: str) -> float:
    """BLEU-1 style: how much of what the answer says is also in the reference."""
    return len(words(answer) & words(reference)) / len(words(answer))


def word_recall(answer: str, reference: str) -> float:
    """ROUGE-1 style: how much of the reference the answer covers."""
    return len(words(answer) & words(reference)) / len(words(reference))


def long_enough(answer: str, min_words: int = 20) -> float:
    return 1.0 if len(answer.split()) >= min_words else 0.0


if __name__ == "__main__":
    print("tool_call_accuracy('search_knowledge_base', 'search_knowledge_base') =",
          tool_call_accuracy("search_knowledge_base", "search_knowledge_base"))
    print("tool_call_accuracy(None, 'get_weather') =", tool_call_accuracy(None, "get_weather"))
    print("keyword_coverage('...generated 900,000 in 2025-Q2', ['Healthcare', '900,000', '2025-Q2']) =",
          round(keyword_coverage("...generated 900,000 in 2025-Q2", ["Healthcare", "900,000", "2025-Q2"]), 3))
    print(f"\n{'answer':<11} exact  keywords  precision  recall  length")
    for name, answer in ANSWERS.items():
        print(f"{name:<11} {exact_match(answer, REFERENCE):<6.1f} {keyword_coverage(answer, KEYWORDS):<9.2f} "
              f"{word_precision(answer, REFERENCE):<10.2f} {word_recall(answer, REFERENCE):<7.2f} "
              f"{long_enough(answer):.1f}")
