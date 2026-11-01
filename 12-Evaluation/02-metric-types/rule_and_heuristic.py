# Rule-based and heuristic metrics in a few lines of plain Python (Chapter 14, Section 14.6).
#
# The examples come from an imaginary travel assistant that searches flights and
# answers baggage questions. Rule-based metrics check a condition the answer must
# meet: the right tool, the required keywords, an exact match. Heuristic metrics
# measure word overlap and length as a stand-in for quality. Two answers to "When
# does flight EK924 leave Cairo?" are scored against "Flight EK924 leaves Cairo at
# 10:15.": one right but reworded, one wrong but worded like the reference.
#
# Prerequisites: none (no model, no API key, standard library only).
# Run it with (from the repository root):
#    python 12-Evaluation/02-metric-types/rule_and_heuristic.py
# Expected result: three rule-based checks, then a table of five scores per answer;
# the word-overlap heuristics score the wrong answer higher than the right one.

import re

REFERENCE = "Flight EK924 leaves Cairo at 10:15."
KEYWORDS = ["EK924", "Cairo", "10:15"]
ANSWERS = {
    "A (right)": "At 10:15, EK924 departs from Cairo.",
    "B (wrong)": "Flight EK924 leaves Dubai at 10:15.",
}


# Rule-based: 1.0 if the agent called the expected tool, else 0.0.
def tool_call_accuracy(actual_tool: str | None, expected_tool: str) -> float:
    return 1.0 if actual_tool == expected_tool else 0.0


# Rule-based: the share of the required keywords that appear in the answer.
def keyword_coverage(answer: str, keywords: list[str]) -> float:
    found = [k for k in keywords if k.lower() in answer.lower()]
    return len(found) / len(keywords)


# Rule-based: 1.0 only if the answer is the reference, character for character.
def exact_match(answer: str, reference: str) -> float:
    return 1.0 if answer.strip() == reference.strip() else 0.0


# Heuristic helper: the distinct words of a text, lowercased, without punctuation.
def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


# Heuristic, BLEU-1 style: how much of what the answer says is also in the reference.
def word_precision(answer: str, reference: str) -> float:
    return len(words(answer) & words(reference)) / len(words(answer))


# Heuristic, ROUGE-1 style: how much of the reference the answer covers.
def word_recall(answer: str, reference: str) -> float:
    return len(words(answer) & words(reference)) / len(words(reference))


# Heuristic: 1.0 if the answer has at least min_words words, a proxy for substance.
def long_enough(answer: str, min_words: int = 20) -> float:
    return 1.0 if len(answer.split()) >= min_words else 0.0


if __name__ == "__main__":
    print("tool_call_accuracy('search_flights', 'search_flights') =",
          tool_call_accuracy("search_flights", "search_flights"))
    print("tool_call_accuracy(None, 'baggage_policy') =", tool_call_accuracy(None, "baggage_policy"))
    print("keyword_coverage('...the fare is 240 USD on 12 March', ['EK924', '240 USD', '12 March']) =",
          round(keyword_coverage("...the fare is 240 USD on 12 March", ["EK924", "240 USD", "12 March"]), 3))
    print(f"\n{'answer':<11} exact  keywords  precision  recall  length")
    for name, answer in ANSWERS.items():
        print(f"{name:<11} {exact_match(answer, REFERENCE):<6.1f} {keyword_coverage(answer, KEYWORDS):<9.2f} "
              f"{word_precision(answer, REFERENCE):<10.2f} {word_recall(answer, REFERENCE):<7.2f} "
              f"{long_enough(answer):.1f}")
