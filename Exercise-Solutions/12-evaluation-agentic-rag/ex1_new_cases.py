# Exercise 14.1 - Grow the golden set: three new cases, graded before they join the dataset.
#
# Two cases come from production questions (Section 14.11); one checks that the
# chatbot admits what the records don't hold. The script runs each through the
# same target and nine evaluators as run_eval.py, without uploading anything, so
# you can see whether a new case passes before you add it.
#
# Prerequisites: .env with OPENAI_API_KEY; the knowledge base ingested in
# 12-Evaluation/04-agentic-rag-evaluation.
# Run it with (from the repository root):
#    python Exercise-Solutions/12-evaluation-agentic-rag/ex1_new_cases.py
# Expected result: one block per case with its nine scores and PASS or FAIL.
# Answers vary between runs; see Appendix B for what we got.

import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "12-Evaluation" / "04-agentic-rag-evaluation"
sys.path.insert(0, str(EVAL_DIR))
os.chdir(EVAL_DIR)  # the app finds .chroma/ and the datasets relative to its folder

from evals.evaluators import ALL_EVALUATORS
from evals.gate import case_passed
from evals.run_eval import target

NEW_CASES = [
    {"id": "sarah_lee_accounts", "query": "Which accounts does Sarah Lee look after?",
     "expected_answer": "Sarah Lee looks after MedCore, AutoWorks and Lumen Bank.",
     "expected_keywords": ["MedCore", "AutoWorks", "Lumen Bank"], "expected_tool": "search_knowledge_base",
     "expected_sources": ["crm_exports/accounts.csv", "crm_exports/sales_reps.csv"]},
    {"id": "fintrust_vs_technova", "query": "Compare FinTrust and TechNova for me.",
     "expected_answer": ("Both are active North American accounts covered by Michael Grant. FinTrust is in "
                         "Finance; its Azure Modernization project is completed. TechNova is in Technology; "
                         "its GraphRAG Platform project is delayed by 35 days."),
     "expected_keywords": ["FinTrust", "TechNova", "Michael Grant"], "expected_tool": "search_knowledge_base",
     "expected_sources": ["crm_exports/accounts.csv", "crm_exports/sales_reps.csv", "projects/projects.csv"]},
    {"id": "technova_phone", "query": "What's TechNova's phone number?",
     "expected_answer": "Says the records don't include TechNova's phone number, without inventing one.",
     "expected_keywords": [], "expected_tool": "search_knowledge_base", "expected_sources": []},
]


# Grade each new case with the same target and evaluators, without adding it to the dataset.
def main() -> None:
    for case in NEW_CASES:
        inputs = {"query": case["query"]}
        outputs = target(inputs)
        scores = {}
        for evaluator in ALL_EVALUATORS:
            params = evaluator.__code__.co_varnames[:evaluator.__code__.co_argcount]
            kwargs = {"inputs": inputs, "outputs": outputs, "reference_outputs": case}
            result = evaluator(**{name: kwargs[name] for name in params})
            scores[result["key"]] = result["score"]
        verdict = "PASS" if case_passed(scores) else "FAIL"
        print(f"{verdict}  {case['id']}\n  answer: {outputs['answer'][:300]!r}")
        print("  " + "  ".join(f"{key} {value:.2f}" for key, value in scores.items()) + "\n")


if __name__ == "__main__":
    main()
