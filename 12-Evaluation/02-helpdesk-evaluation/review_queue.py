# Close the loop, part 1: send flagged production runs to a LangSmith annotation queue.
#
# This script reads recent root runs from the "corniche-cycles-prod" project
# (written by online_feedback.py), keeps the ones where any feedback score is
# below 1 - a thumbs down from a customer or a failed automatic check - and adds
# them to the annotation queue "corniche-cycles-review", creating it the first
# time with instructions for the reviewer.
#
# Why this matters:
# - An annotation queue is LangSmith's inbox for HUMAN evaluation: a reviewer
#   opens each run with its full trace, grades it, and can add it to a dataset
#   with a corrected reference. That's the fourth kind of evaluation, and the
#   only one that can decide what "correct" means for a question nobody planned.
# - The script stops at the queue on purpose. A production reply is not a
#   correct answer; copying flagged runs straight into the golden set would teach
#   your regression gate to defend the mistake.
#
# Before you run this:
# 1. Run online_feedback.py first, so the project has runs with feedback.
# 2. LANGSMITH_API_KEY in a .env file in this folder or the repository root.
#
# Run it with (from the 12-Evaluation/02-helpdesk-evaluation folder):
#    python review_queue.py
#
# Expected result: three lines, one per flagged question (opening hours, return
# station, 3 electric bikes), then "queued 3 runs in corniche-cycles-review".
# Running it again adds the same runs again; that's harmless.

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from langsmith import Client

PROJECT = "corniche-cycles-prod"
QUEUE = "corniche-cycles-review"
INSTRUCTIONS = ("For each run, decide what the help desk SHOULD have done. If it's a case the "
                "golden set should cover, add it to the corniche-cycles-desk dataset with "
                "expected_calls, must_include and expected_behavior filled in.")


def flagged(run) -> list[str]:
    """The feedback keys on this run whose average score is below 1."""
    stats = run.feedback_stats or {}
    return [key for key, s in stats.items() if s["avg"] is not None and s["avg"] < 1]


def main():
    client = Client()
    queue = next(iter(client.list_annotation_queues(name=QUEUE)), None) or \
        client.create_annotation_queue(name=QUEUE, rubric_instructions=INSTRUCTIONS)
    picked = {}
    for run in client.list_runs(project_name=PROJECT, is_root=True, limit=50):
        problems = flagged(run)
        question = run.inputs["messages"][0]["content"]
        if problems and question not in picked:  # newest run per question
            picked[question] = run.id
            print(f"{question[:50]:<50} {problems}")
    client.add_runs_to_annotation_queue(queue.id, run_ids=list(picked.values()))
    print(f"queued {len(picked)} runs in {QUEUE}")


if __name__ == "__main__":
    main()
