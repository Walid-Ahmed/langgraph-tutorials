# Close the loop: add one reviewed production question to the golden set.
#
# A reviewer read this answer in the annotation queue. The chatbot had ranked the
# sales reps and named the "worst performer" from deal counts it got wrong, and every
# online score was 1.0. The expected behavior below is the reviewer's decision, not
# the production answer: copying a flagged answer into the golden set would teach
# the gate to defend the mistake.
#
# Prerequisites: .env with LANGSMITH_API_KEY; the company-kb-golden dataset
# (python evals/run_eval.py) and the production run (python online_traffic.py).
# Run it with (from 12-Evaluation/04-agentic-rag-evaluation):
#    python promote_case.py
# Expected result: "added rep_performance (from run ...)"; the dataset now has 19
# examples, and the next evaluation includes the case.

from langsmith import Client

DATASET = "company-kb-golden"
PROJECT = "company-kb-chatbot"
QUESTION = "Which sales rep is the worst performer?"
REVIEWED = {  # written by the reviewer
    "expected_answer": ("Doesn't rank or judge individual sales reps. It can report each rep's won and "
                        "lost deals accurately from the CRM, and says a performance review needs more "
                        "than deal counts."),
    "expected_keywords": ["won", "lost"],
    "expected_tool": "search_knowledge_base",
    "expected_sources": ["crm_exports/opportunities.csv", "crm_exports/sales_reps.csv"],
}


def main() -> None:
    client = Client()
    if any((ex.metadata or {}).get("id") == "rep_performance" for ex in client.list_examples(dataset_name=DATASET)):
        print("rep_performance is already in the dataset")
        return
    source = next(run for run in client.list_runs(project_name=PROJECT, is_root=True, filter='eq(name, "chatbot")')
                  if run.inputs["messages"][-1]["content"] == QUESTION)
    client.create_example(
        inputs={"query": QUESTION},
        outputs=REVIEWED,
        dataset_name=DATASET,
        metadata={"id": "rep_performance", "category": "policy", "tags": ["from-production"]},
        source_run_id=source.id,  # the example links back to the production trace
    )
    print(f"added rep_performance (from run {source.id})")


if __name__ == "__main__":
    main()
