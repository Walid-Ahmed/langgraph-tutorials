# Offline evaluation of the agentic RAG chatbot, and the CI gate.
#
# The whole run, in five steps:
#   1. run_ingestion()      make sure Chroma holds the company files
#   2. ensure_dataset()     upload golden_dataset.json to LangSmith the first time
#   3. client.evaluate()    ask the agent every golden question; score 9 metrics each
#   4. summarize()          averages and pass rate; a "passed" score on every case's trace
#   5. latest.json          per-case results + summary saved locally, beside the LangSmith experiment
#
# Why this matters:
# - The target calls ask_with_tools, the same function the chat UI uses, so the
#   evaluation measures the code path users hit.
# - The experiment in LangSmith links every score to the trace that produced it;
#   results/latest.json keeps the same results in the repository for CI and diffs.
# - The exit code is the CI integration: 1 when fewer than 80% of cases pass, or
#   when a case that passed in the reviewed baseline fails now.
# - One run makes about 40 gpt-4o-mini calls and 36 gpt-4o judge calls: a few US cents.
#
# Prerequisites: .env with OPENAI_API_KEY, LANGSMITH_API_KEY and LANGSMITH_TRACING=true,
# the datasets/ folder, and `pip install -r requirements.txt` from the repository root.
# Run it with (from 12-Evaluation/04-agentic-rag-evaluation):
#    python evals/run_eval.py                  # evaluate; compare with the baseline if one exists
#    python evals/run_eval.py --set-baseline   # after reviewing a run, keep it as the baseline
# Expected result: one line per case, then "Pass rate: ..%" and the average of each metric.

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))  # so "src" and "evals" import when run as a script

from langsmith import Client

from evals.evaluators import ALL_EVALUATORS
from evals.gate import MIN_PASS_RATE, case_passed, find_regressions, summarize
from src import config
from src.agents import ask_with_tools
from src.pipeline import run_ingestion

GOLDEN_DATASET_PATH = ROOT_DIR / "evals" / "data" / "golden_dataset.json"
RESULTS_DIR = ROOT_DIR / "evals" / "results"
DATASET = "company-kb-golden"

client = Client()


# Upload the golden set once; after that the LangSmith dataset is the one that counts.
def ensure_dataset() -> None:
    if client.has_dataset(dataset_name=DATASET):
        return
    cases = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))["cases"]
    dataset = client.create_dataset(DATASET, description="Company knowledge-base chatbot golden set")
    client.create_examples(dataset_id=dataset.id, examples=[{
        "inputs": {"query": case["query"]},
        "outputs": {k: case[k] for k in ("expected_answer", "expected_keywords",
                                         "expected_tool", "expected_sources")},
        "metadata": {"id": case["id"], "category": case["category"]},
    } for case in cases])
    print(f"created dataset {DATASET} with {len(cases)} examples")


# The system under test: one golden question in, what the agent did out.
def target(inputs: dict) -> dict:
    reply = ask_with_tools(inputs["query"], online_eval=False)
    return {"answer": reply.answer, "tool": reply.tools[0] if reply.tools else None,
            "tools": reply.tools, "sources": reply.sources, "contexts": reply.contexts}


# The whole offline run: ingest, upload the golden set, evaluate, apply the gate, save the results.
def run(set_baseline: bool) -> int:
    run_ingestion()
    ensure_dataset()
    results = client.evaluate(
        target, data=DATASET, evaluators=ALL_EVALUATORS,
        experiment_prefix="kb-agent",
        metadata={"agent_model": config.AGENT_MODEL, "judge_model": config.JUDGE_MODEL},
        max_concurrency=4,
    )
    cases = []
    for row in results:
        example, run_ = row["example"], row["run"]
        if run_ is None or run_.error:
            raise SystemExit(f"{example.metadata['id']}: the agent failed on this case")
        scores = {r.key: r.score for r in row["evaluation_results"]["results"]}
        passed = case_passed(scores)
        client.create_feedback(run_.id, key="passed", score=int(passed))
        cases.append({
            "id": example.metadata["id"], "category": example.metadata["category"],
            "query": example.inputs["query"], "expected_answer": example.outputs["expected_answer"],
            "actual_answer": run_.outputs["answer"], "actual_tool": run_.outputs["tool"],
            "retrieved_sources": run_.outputs["sources"], "scores": scores, "passed": passed,
        })
        print(f"{'PASS' if passed else 'FAIL'}  {cases[-1]['id']:<32} "
              f"correctness {scores['correctness']:.1f}  tool {cases[-1]['actual_tool']}")
    cases.sort(key=lambda case: case["id"])
    summary = summarize(cases)

    RESULTS_DIR.mkdir(exist_ok=True)
    latest = RESULTS_DIR / "latest.json"
    latest.write_text(json.dumps({"experiment": results.experiment_name, "summary": summary,
                                  "cases": cases}, indent=2), encoding="utf-8")
    print(f"\nSaved {len(cases)} results to {latest.relative_to(ROOT_DIR)}")
    print(f"Pass rate: {summary['pass_rate']:.0%} ({summary['passed_cases']}/{summary['total_cases']})")
    print(f"Average scores: {summary['average_scores']}")

    failed = False
    if summary["pass_rate"] < MIN_PASS_RATE:
        print(f"FAILED: pass rate {summary['pass_rate']:.0%} is below the required {MIN_PASS_RATE:.0%}.")
        failed = True
    baseline = RESULTS_DIR / "baseline.json"
    if baseline.exists():
        regressed = find_regressions(json.loads(baseline.read_text(encoding="utf-8"))["cases"], cases)
        for case_id in regressed:
            print(f"REGRESSED {case_id}: passed in the baseline, fails now")
        failed = failed or bool(regressed)
    if set_baseline:
        shutil.copyfile(latest, baseline)
        print(f"Baseline set to this run ({results.experiment_name})")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation and CI gate")
    parser.add_argument("--set-baseline", action="store_true",
                        help="keep this run as the baseline future runs are compared with")
    sys.exit(run(parser.parse_args().set_baseline))
