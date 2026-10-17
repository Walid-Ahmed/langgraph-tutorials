# Run from the repository root:
#   python Exercise-Solutions/12-evaluation/ex4_close_the_loop.py
#
# What it does:
#   Exercise 14.4, close the loop. On a copy of the chapter's dataset
#   ("corniche-cycles-desk-ex14-4", 13 examples including group-quote), it:
#     1. wraps grounded_numbers from online_feedback.py as a fourth experiment
#        evaluator and runs v2 with all four, the baseline: group-quote fails
#        grounded_numbers, because the desk multiplies 150 by 3 itself;
#     2. registers a new desk version, v4: v2's prompt with a rental_quote that
#        takes a `quantity` argument (12-Evaluation/desk.py is not edited);
#     3. runs the chapter's regression gate (regression_gate.gate) for v2 -> v4,
#        grading the candidate with the same four evaluators.
#   It prints group-quote's route, tool results and reply for both versions.
#
# Why it matters:
#   - grounded_numbers needs no golden answer, so the same function grades live
#     traffic (Section 14.9) and the golden set. Here it turns the production
#     flag into a failing offline test that every future version must pass.
#   - The fix goes where the bug is: the desk had no way to price a group, so the
#     model did the arithmetic itself, against its own rule. With a quantity
#     argument, the tool does the pricing, discount included.
#   - The gate checks that the fix turns group-quote green without turning
#     anything else red.
#
# Prerequisites:
#   pip install -r requirements.txt, then OPENAI_API_KEY, LANGSMITH_API_KEY and
#   LANGSMITH_TRACING=true in the repo-root .env or the environment. The chapter's
#   dataset must already hold group-quote (run promote_example.py in 12-Evaluation).
#   Two experiments over 13 examples: about 55 gpt-4o-mini calls and 26 gpt-4o
#   judge calls, a few US cents.
#
# Expected result (our run):
#   v2: correct_tools 1.00 | facts_present 1.00 | judge_reply 1.00 | grounded_numbers 0.92
#   group-quote: tool result "electric bike, 2 h: EGP 150", reply "... EGP 450"
#     grounded_numbers=0: numbers from nowhere: ['450']
#   improved  group-quote     grounded_numbers0.00 -> 1.00
#   PASS: v4 is no worse than v2 on any example
#   group-quote under v4: tool result "3 electric bikes, 2 h: EGP 450"
# and exit status 0. The wording of the replies varies between runs.

import os
import sys

from dataset_copy import ensure_copy, show_replies, use_dataset

import desk  # noqa: E402
import regression_gate  # noqa: E402
from run_experiment import client, run  # noqa: E402  (loads .env)
from evaluators import correct_tools, facts_present, judge_reply  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

# Importing online_feedback sends traces to the production project (it sets
# LANGSMITH_PROJECT). Put the setting back, so nothing here lands in production.
project_before = os.environ.get("LANGSMITH_PROJECT")
from online_feedback import grounded_numbers as check_grounded  # noqa: E402

if project_before is None:
    os.environ.pop("LANGSMITH_PROJECT", None)
else:
    os.environ["LANGSMITH_PROJECT"] = project_before

COPY = "corniche-cycles-desk-ex14-4"


def grounded_numbers(inputs: dict, outputs: dict) -> dict:
    """Reference-free: 1 if every number in the reply came from the question or a tool."""
    result = check_grounded(inputs["question"], outputs)
    return {"score": result["score"], "comment": result["reason"]}


EVALUATORS = (correct_tools, facts_present, judge_reply, grounded_numbers)
METRICS = [e.__name__ for e in EVALUATORS]  # LangSmith names each feedback after its function


@tool("rental_quote")  # same tool name, so the golden set's expected_calls still match
def rental_quote_group(bike_type: desk.BikeType, hours: int, quantity: int = 1) -> str:
    """Price a rental of one or more bikes of the same type, in Egyptian pounds (EGP).
    Applies the long-rental discount."""
    base = desk.HOURLY_RATE[bike_type] * hours * quantity
    bikes = f"{bike_type} bike" if quantity == 1 else f"{quantity} {bike_type} bikes"
    if hours >= desk.LONG_RENTAL_HOURS:
        total = round(base * (1 - desk.LONG_RENTAL_DISCOUNT))
        return f"{bikes}, {hours} h: EGP {total} (EGP {base} minus 20% long-rental discount)"
    return f"{bikes}, {hours} h: EGP {base}"


# v4 = v2's prompt with the quantity-aware tool. build_desk() looks versions up
# in desk.VERSIONS, so registering it there is all the "update" the desk needs.
v2_prompt, _ = desk.VERSIONS["v2"]
desk.VERSIONS["v4"] = (v2_prompt, [rental_quote_group, desk.bikes_available])


def print_scores(version: str, scores: dict) -> None:
    print(f"{'example':<16}" + "".join(f"{m[:8]:>10}" for m in METRICS))
    for example_id, s in sorted(scores.items()):
        print(f"{example_id:<16}" + "".join(f"{s[m]:>10.2f}" for m in METRICS))
    averages = {m: sum(s[m] for s in scores.values()) / len(scores) for m in METRICS}
    print(f"\n{version}: " + " | ".join(f"{m} {v:.2f}" for m, v in averages.items()))


def main() -> int:
    print(f"{COPY}: {ensure_copy(client, COPY)} examples\n")
    use_dataset(COPY)

    # 1. The baseline: v2 graded by all four evaluators, stored as an experiment.
    print_scores("v2", run("v2", evaluators=EVALUATORS))
    # show_replies also waits until the baseline's feedback can be read back from
    # LangSmith, which the gate below depends on.
    show_replies(client, COPY, "v2", {"group-quote"}, METRICS)

    # 2-3. The gate, with the candidate graded by the same four evaluators.
    print()
    status = regression_gate.gate("v2", "v4", metrics=tuple(METRICS), evaluators=EVALUATORS)
    show_replies(client, COPY, "v4", {"group-quote"}, METRICS)
    return status


if __name__ == "__main__":
    sys.exit(main())
