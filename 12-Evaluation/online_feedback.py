# Online evaluation: score live traffic, where there is no golden answer.
#
# This script plays the part of the help desk in production. It answers a few
# customer questions that are NOT in the golden set, traces each run to the
# "corniche-cycles-prod" LangSmith project under a run id we choose, and then
# attaches two kinds of feedback to those runs:
#   user_thumbs       the customer's thumbs up/down (simulated here; in a real
#                     app it comes from a button next to the reply)
#   grounded_numbers  an automatic, reference-free check: every number in the
#                     reply must appear in the question or in a tool result
#
# Why this matters:
# - Online there is no expected answer to compare against, so online evaluators
#   are REFERENCE-FREE: they check properties any good reply has (grounded,
#   polite, on-policy) instead of matching a stored answer.
# - Choosing the run id up front (config={"run_id": ...}) is what lets feedback
#   that arrives later - a click, a nightly check - land on the right trace.
# - Users and automatic checks catch different failures. Below, the customers
#   flag a scope gap the golden set never covered; the grounding check flags an
#   answer the customer was happy with.
#
# Before you run this:
# 1. pip install -r requirements.txt   (from the repository root)
# 2. A .env file in this folder or the repository root with OPENAI_API_KEY,
#    LANGSMITH_API_KEY and LANGSMITH_TRACING=true.
#
# Run it with (from the 12-Evaluation folder):
#    python online_feedback.py
#
# Expected result: four lines, one per question, each showing the reply and its
# two scores. On our run "3 electric bikes" got user 1 / grounded 0 (the desk
# multiplied 150 x 3 itself), and the opening-hours and return-station questions
# got user 0 / grounded 1. Then: "feedback attached to 4 runs".

import os
import re
import uuid

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
os.environ["LANGSMITH_PROJECT"] = "corniche-cycles-prod"  # production traces go here

from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from desk import build_desk, run_desk

# (question, simulated thumbs: 1 up / 0 down, the customer's comment)
LIVE_TRAFFIC = [
    ("city bike 2 hrs how much", 1, ""),
    ("I need 3 electric bikes for 2 hours for my family. Total?", 1, ""),
    ("Is Stanley Bridge open at 6 a.m.?", 0, "It's a rental question! I just want the opening time."),
    ("Can I return the bike at a different station?", 0, "Why can't it answer this?"),
]


def grounded_numbers(question: str, outputs: dict) -> dict:
    """Reference-free: 1 if every number in the reply came from the question or a tool."""
    evidence = " ".join([question, *outputs["tool_results"]])
    numbers = re.findall(r"\d+", outputs["answer"].replace(",", ""))
    unsupported = [n for n in numbers if not re.search(rf"\b{n}\b", evidence)]
    if unsupported:
        return {"score": 0, "reason": f"numbers from nowhere: {unsupported}"}
    return {"score": 1, "reason": "every number is backed by the question or a tool"}


def main():
    client = Client()
    desk = build_desk("v2")
    served = []
    for question, thumbs, comment in LIVE_TRAFFIC:
        run_id = uuid.uuid4()
        outputs = run_desk(desk, question, config={
            "run_id": run_id, "tags": ["production"], "metadata": {"version": "v2"},
        })
        served.append((run_id, question, outputs, thumbs, comment))

    wait_for_all_tracers()  # make sure the traces are uploaded before we annotate them

    for run_id, question, outputs, thumbs, comment in served:
        check = grounded_numbers(question, outputs)
        client.create_feedback(run_id, key="user_thumbs", score=thumbs, comment=comment or None)
        client.create_feedback(run_id, key="grounded_numbers", score=check["score"],
                               comment=check["reason"])
        print(f"user {thumbs} | grounded {check['score']} | {question[:42]:<42} | {check['reason']}")
    print(f"feedback attached to {len(served)} runs")


if __name__ == "__main__":
    main()
