# Close the loop: turn flagged production runs into candidate golden examples.
#
# This script reads recent root runs from the "corniche-cycles-prod" LangSmith
# project, keeps the ones where any feedback score is below 1 (a thumbs down
# from a customer, or a failed automatic check), and writes each one to
# review_queue.jsonl with its question, the desk's reply and the feedback.
# The expected fields are left as "TODO" for a person to fill in.
#
# Why this matters:
# - This is the arrow from online back to offline. A failure a customer found
#   once becomes a test every future version has to pass.
# - The script stops at a REVIEW QUEUE on purpose. A production reply is not a
#   correct answer, and only a person can write the expected behavior. Copying
#   flagged runs straight into golden_set.jsonl would fill it with mistakes.
#
# Before you run this:
# 1. Run online_feedback.py first, so the project has runs with feedback.
# 2. LANGSMITH_API_KEY in a .env file in this folder or the repository root.
#
# Run it with (from the 12-Evaluation folder):
#    python harvest_failures.py
#
# Expected result: three lines, one per flagged question (opening hours, return
# station, 3 electric bikes), then "wrote 3 candidates to review_queue.jsonl".

import json
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from langsmith import Client

PROJECT = "corniche-cycles-prod"


def flagged(run) -> dict:
    """The feedback keys on this run whose average score is below 1, with their comments."""
    stats = run.feedback_stats or {}
    return {key: s["comments"] for key, s in stats.items() if s["avg"] is not None and s["avg"] < 1}


def main():
    client = Client()
    candidates = {}
    for run in client.list_runs(project_name=PROJECT, is_root=True, limit=50):
        problems = flagged(run)
        if not problems:
            continue
        question = run.inputs["messages"][0]["content"]
        candidates.setdefault(question, {  # one entry per question, newest run first
            "question": question,
            "reply_in_production": run.outputs["messages"][-1]["content"],
            "feedback": problems,
            "trace_id": str(run.trace_id),
            "expected_tools": "TODO", "must_include": "TODO", "expected_behavior": "TODO",
        })
    with open("review_queue.jsonl", "w") as f:
        for item in candidates.values():
            f.write(json.dumps(item) + "\n")
            print(f"{item['question'][:50]:<50} {list(item['feedback'])}")
    print(f"wrote {len(candidates)} candidates to {Path('review_queue.jsonl')}")


if __name__ == "__main__":
    main()
