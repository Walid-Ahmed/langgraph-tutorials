# Run from the repository root:
#   python Exercise-Solutions/12-evaluation/ex1_grow_golden_set.py
#   python Exercise-Solutions/12-evaluation/ex1_grow_golden_set.py v1   (any desk version)
#
# What it does:
#   Exercise 14.1. Copies the chapter's "corniche-cycles-desk" dataset to
#   "corniche-cycles-desk-ex14-1", adds the three examples in
#   ex1_new_golden_examples.jsonl to the copy, and runs the chapter's experiment
#   (run_experiment.run) for v2 of the desk over every example. The three new
#   examples are:
#     style-typo    all lowercase, text-speak, and "stanly bridge" for Stanley Bridge
#     stock-zero    a kids' bike at Bibliotheca, where the stock is 0
#     price-4-vs-5  a city bike for 4 hours versus 5 hours (both cost EGP 160)
#   It prints the usual score table, then the route, tool results, reply and
#   evaluator comments of the three new examples, read back from LangSmith.
#
# Why it matters:
#   - A new golden example earns its place by covering something the set didn't:
#     a different writing style, a stock of zero, and the discount boundary seen
#     from the customer's side (the fifth hour is free, so a desk that did its
#     own arithmetic could easily get it wrong).
#   - The experiment runs over the WHOLE dataset, not just the new lines, so a
#     drop in score can only come from the new examples.
#   - Adding them to a copy keeps the chapter's dataset, and every number the
#     chapter reports for it, unchanged.
#
# Prerequisites:
#   pip install -r requirements.txt, then OPENAI_API_KEY, LANGSMITH_API_KEY and
#   LANGSMITH_TRACING=true in the repo-root .env or the environment. The chapter's
#   dataset must exist (run 12-Evaluation/02-helpdesk-evaluation/run_experiment.py once).
#   One experiment over 16 examples: about 35 gpt-4o-mini calls and 16 gpt-4o
#   judge calls, a few US cents.
#
# Expected result: style-typo fails, the other two new examples pass, and so do
# the 13 chapter examples. Our two runs of v2 over 16 examples failed style-typo
# in two different ways:
#   run 1: v2: correct_tools 1.00 | facts_present 0.94 | judge_reply 0.94
#          one bikes_available("stanly bridge") call, "Unknown station", and a
#          reply asking the customer to confirm the station (facts 0, judge 0)
#   run 2: v2: correct_tools 0.94 | facts_present 1.00 | judge_reply 1.00
#          a second call with "Stanley Bridge" and a correct "6 city bikes"
#          reply, but one tool call more than expected (tools 0)
# The wording of the replies varies between runs.

import sys
from pathlib import Path

from dataset_copy import ensure_copy, load_jsonl, show_replies, use_dataset

from run_experiment import client, print_table, run  # noqa: E402  (loads .env first)

COPY = "corniche-cycles-desk-ex14-1"
NEW_EXAMPLES = load_jsonl(Path(__file__).parent / "ex1_new_golden_examples.jsonl")


def main(version: str):
    size = ensure_copy(client, COPY, NEW_EXAMPLES)
    print(f"{COPY}: {size} examples\n")
    use_dataset(COPY)
    print_table(version, run(version))
    # Read the new examples' replies, not only their scores: a pass for the
    # wrong reason, or a fail for a good one, is worth knowing about.
    show_replies(client, COPY, version, {ex["id"] for ex in NEW_EXAMPLES},
                 metrics=["correct_tools", "facts_present", "judge_reply"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v2")
