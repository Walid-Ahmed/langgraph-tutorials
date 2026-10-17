# Close the loop, part 2: add one reviewed production case to the golden dataset.
#
# A reviewer looked at the "3 electric bikes" run in the annotation queue and
# decided what the desk should have done. This script records that decision as a
# new example in the corniche-cycles-desk dataset, linked to the production run
# it came from. (In the UI the reviewer can do the same with "Add to dataset".)
#
# Why this matters:
# - The question comes from a real customer; the expected outputs come from a
#   person. That pairing is what makes a production failure a trustworthy test.
# - source_run_id keeps the link: from the example you can open the original
#   production trace, months later.
# - LangSmith versions the dataset on every change, and each experiment records
#   the dataset version it ran on, so older experiments stay comparable.
#
# Before you run this:
# 1. Run online_feedback.py and review_queue.py first.
# 2. LANGSMITH_API_KEY in a .env file in this folder or the repository root.
#
# Run it with (from the 12-Evaluation folder):
#    python promote_example.py
#
# Expected result: "added group-quote (from run <id>)" the first time, and
# "group-quote is already in the dataset" after that. The next run_experiment.py
# run then has 13 examples instead of 12.

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from langsmith import Client

DATASET = "corniche-cycles-desk"
QUESTION = "I need 3 electric bikes for 2 hours for my family. Total?"
REVIEWED = {  # written by the reviewer, not copied from the production reply
    "expected_calls": [{"name": "rental_quote", "args": {"bike_type": "electric", "hours": 2}}],
    "must_include": ["450"],
    "expected_behavior": "Quotes EGP 450 for three electric bikes for 2 hours.",
}


def main():
    client = Client()
    if any(ex.metadata.get("id") == "group-quote" for ex in client.list_examples(dataset_name=DATASET)):
        print("group-quote is already in the dataset")
        return
    source = next(r for r in client.list_runs(project_name="corniche-cycles-prod", is_root=True)
                  if r.inputs["messages"][0]["content"] == QUESTION)
    client.create_example(
        inputs={"question": QUESTION},
        outputs=REVIEWED,
        dataset_name=DATASET,
        metadata={"id": "group-quote", "tags": ["pricing", "group", "from-production"]},
        source_run_id=source.id,
    )
    print(f"added group-quote (from run {source.id})")


if __name__ == "__main__":
    main()
