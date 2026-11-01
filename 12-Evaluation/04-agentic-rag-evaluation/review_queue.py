# Human review: send live answers to a LangSmith annotation queue.
#
# A run goes to the queue for one of two reasons: an online score fell below
# FLAG_BELOW, or it was drawn in a small random sample of the runs nobody flagged.
# The sample matters because the online checks only see what they were written to
# see: an answer can be polite, relevant and still wrong, or inappropriate.
#
# Prerequisites: .env with LANGSMITH_API_KEY, and production traces with online
# scores in the "company-kb-chatbot" project (python online_traffic.py).
# Run it with (from 12-Evaluation/04-agentic-rag-evaluation):
#    python review_queue.py
# Expected result: one line per queued run, with why it was chosen, then
# "queued N runs in company-kb-review". Open the queue in the LangSmith UI to review.

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langsmith import Client

PROJECT = "company-kb-chatbot"
QUEUE = "company-kb-review"
ONLINE_KEYS = ("toxicity", "bias_fairness", "answer_relevancy")
FLAG_BELOW = 0.8      # an online score under this sends the run to a person
SAMPLE_SIZE = 2       # unflagged runs a person reads anyway
INSTRUCTIONS = ("Decide what the chatbot SHOULD have answered. If the case belongs in the golden "
                "set, write its expected answer, keywords, tool and sources, and add it to "
                "company-kb-golden with promote_case.py.")


def select_for_review(runs: list[dict], flag_below: float = FLAG_BELOW, sample_size: int = SAMPLE_SIZE,
                      seed: int = 14) -> list[tuple[dict, str]]:
    """Every run with a low online score, plus a seeded random sample of the rest, with the reason."""
    flagged, unflagged = [], []
    for run in runs:
        low = [key for key in ONLINE_KEYS if run["scores"].get(key, 1.0) < flag_below]
        (flagged if low else unflagged).append((run, f"flagged: {', '.join(low)}" if low else "sampled"))
    sampled = random.Random(seed).sample(unflagged, min(sample_size, len(unflagged)))
    return flagged + sampled


def latest_runs(client: Client) -> list[dict]:
    """The newest root run per question in the production project, with its online scores."""
    runs = {}
    # Newest first. Only the chatbot's own runs: the online judge's calls are traced here too.
    for run in client.list_runs(project_name=PROJECT, is_root=True, filter='eq(name, "chatbot")', limit=100):
        question = run.inputs["messages"][-1]["content"]
        if question not in runs:
            stats = run.feedback_stats or {}
            runs[question] = {"id": run.id, "question": question,
                              "scores": {key: stats[key]["avg"] for key in ONLINE_KEYS if key in stats}}
    return list(runs.values())


def main() -> None:
    client = Client()
    queue = next(iter(client.list_annotation_queues(name=QUEUE)), None) or \
        client.create_annotation_queue(name=QUEUE, rubric_instructions=INSTRUCTIONS)
    chosen = select_for_review(latest_runs(client))
    for run, reason in chosen:
        print(f"{reason:<28} {run['question']}")
    client.add_runs_to_annotation_queue(queue.id, run_ids=[run["id"] for run, _ in chosen])
    print(f"queued {len(chosen)} runs in {QUEUE}")


if __name__ == "__main__":
    main()
