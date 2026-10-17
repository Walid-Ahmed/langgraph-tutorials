# Run from the repository root:
#   python Exercise-Solutions/12-evaluation/ex3_route_only_gate.py
#
# What it does:
#   Exercise 14.3, break it on purpose. Runs the chapter's regression gate
#   (regression_gate.gate) for v2 -> v3, but with the candidate graded by
#   correct_tools ONLY. This is the exercise's one-line edit,
#       new = run(candidate, evaluators=(correct_tools,))
#   made from outside, so 12-Evaluation/regression_gate.py is never edited and
#   there is no line to forget to put back.
#
# Why it matters:
#   - v3's only change is an off-by-one in the pricing tool (">=" became ">"), so
#     a 5-hour rental loses its discount. The desk still calls rental_quote with
#     the right arguments; the TOOL returns the wrong price.
#   - The gate compares only the metrics that both experiments have. With the
#     candidate graded on the route alone, the price-boundary regression in
#     facts_present and judge_reply is never looked at, and the gate passes.
#   - A regression gate is exactly as good as the examples AND the metrics
#     behind it.
#
# Prerequisites:
#   pip install -r requirements.txt, then OPENAI_API_KEY, LANGSMITH_API_KEY and
#   LANGSMITH_TRACING=true in the repo-root .env or the environment, and a stored
#   v2 baseline (run `python run_experiment.py v2` in 12-Evaluation once).
#   About 30 gpt-4o-mini calls and no judge calls.
#
# Expected result (our run):
#   baseline: desk-v2-<suffix>
#   PASS: v3 is no worse than v2 on any example
# with no REGRESSED lines, and exit status 0, so a CI job would let v3 merge.
# The unmodified gate, `python regression_gate.py v2 v3`, prints the price-boundary
# regression and exits 1.

import functools
import sys

from dataset_copy import EVAL_DIR  # noqa: F401  (puts 12-Evaluation on the import path)

import regression_gate  # noqa: E402  (loads .env)
from evaluators import correct_tools  # noqa: E402

# gate() calls run(candidate); this makes that call grade the route only.
regression_gate.run = functools.partial(regression_gate.run, evaluators=(correct_tools,))

if __name__ == "__main__":
    sys.exit(regression_gate.gate("v2", "v3"))
