# Read one golden case back from LangSmith, to see how the upload stored it.
#
# run_eval.py uploads golden_dataset.json once (ensure_dataset). Each case becomes
# an example in three parts: inputs (what the target receives), reference outputs
# (what the evaluators compare against) and metadata (everything else). LangSmith
# adds a dataset_split of its own and may store the keys in a different order.
#
# Prerequisites: .env with LANGSMITH_API_KEY, and the company-kb-golden dataset
# (python evals/run_eval.py has run at least once).
# Run it with (from 12-Evaluation/04-agentic-rag-evaluation):
#    python show_example.py
# Expected result: the snowflake_sales_rep case's inputs, reference outputs and metadata.

from pprint import pprint

from dotenv import find_dotenv, load_dotenv
from langsmith import Client

load_dotenv(find_dotenv(usecwd=True))

DATASET = "company-kb-golden"
CASE_ID = "snowflake_sales_rep"


# Find one example by the id the golden set gave it, and print its three parts.
def main() -> None:
    example = next(Client().list_examples(dataset_name=DATASET, metadata={"id": CASE_ID}))
    for part, value in [("inputs", example.inputs), ("outputs", example.outputs),
                        ("metadata", example.metadata)]:
        print(f"{part}:")
        pprint(value, width=96, sort_dicts=False)


if __name__ == "__main__":
    main()
