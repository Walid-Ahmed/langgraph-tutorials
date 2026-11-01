# Exercise 14.2 - Judge the judge: how often does each judge agree with a person?
#
# ex2_run1_answers.json holds the chatbot's 18 answers from the chapter's first run.
# HAND_LABELS is our own pass/fail reading of them: two are wrong (an ID instead of
# a name; a lost deal listed as won). The script grades the same answers with the
# chapter's correctness judge on gpt-4o and on gpt-4o-mini, and counts agreement.
# Grading stored answers means only the judge changes between the two columns.
#
# Prerequisites: .env with OPENAI_API_KEY.
# Run it with (from the repository root):
#    python Exercise-Solutions/12-evaluation-agentic-rag/ex2_judge_agreement.py
# Expected result: a table of 18 answers with both judges' scores, then each judge's
# agreement with the hand labels. Judge scores can vary slightly between runs.

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "12-Evaluation" / "04-agentic-rag-evaluation"))

from langchain_openai import ChatOpenAI

from evals.gate import PASS_THRESHOLD
from evals.metrics import judges

HAND_LABELS = {"snowflake_sales_rep": False, "won_opportunities": False}  # every other answer is right


def grade_with(model: str, answer: dict) -> float:
    judge = ChatOpenAI(model=model, temperature=0).with_structured_output(judges.Grade)
    prompt = (f"Question:\n{answer['query']}\n\nExpected answer:\n{answer['expected_answer']}\n\n"
              f"Answer to grade:\n{answer['actual_answer']}")
    grade = judge.invoke([("system", judges.CORRECTNESS_STEPS), ("human", prompt)])
    assert isinstance(grade, judges.Grade)
    return grade.score / 10


def main() -> None:
    answers = json.loads((HERE / "ex2_run1_answers.json").read_text(encoding="utf-8"))
    models = ["gpt-4o", "gpt-4o-mini"]
    with ThreadPoolExecutor(8) as pool:
        grades = {m: list(pool.map(lambda a, m=m: grade_with(m, a), answers)) for m in models}
    agree = dict.fromkeys(models, 0)
    print(f"{'case':<32} {'hand':<5} {'gpt-4o':>7} {'4o-mini':>8}")
    for i, answer in enumerate(answers):
        hand = HAND_LABELS.get(answer["id"], True)
        row = []
        for model in models:
            passed = grades[model][i] >= PASS_THRESHOLD
            agree[model] += passed == hand
            row.append(f"{grades[model][i]:.1f}{'' if passed == hand else '*'}")
        print(f"{answer['id']:<32} {'pass' if hand else 'fail':<5} {row[0]:>7} {row[1]:>8}")
    for model in models:
        print(f"{model}: agrees with the hand labels on {agree[model]}/{len(answers)}")
    print("* = the judge's pass/fail differs from the hand label")


if __name__ == "__main__":
    main()
