# Offline evaluation: replay the golden dataset through the help desk and score it.
#
# This script loads golden_set.jsonl, runs every question through one version of
# the help desk (v1, v2 or v3 - see desk.py), applies the three evaluators from
# evaluators.py, prints one row per example plus the average of each metric,
# and saves everything to results/<version>.json for compare_runs.py.
#
# Why this matters:
# - This is the whole idea of offline evaluation in one loop: a FIXED dataset,
#   the system under test, and scorers. No live users, no LangSmith account.
# - Saving per-example results (not just averages) is what makes a regression
#   check possible later: you can say WHICH example broke, not only that the
#   average dropped.
# - One run makes about 25 desk calls (gpt-4o-mini) and 12 judge calls (gpt-4o);
#   together they cost a few US cents.
#
# Before you run this:
# 1. pip install -r requirements.txt   (from the repository root)
# 2. OPENAI_API_KEY in a .env file in this folder or the repository root.
#
# Run it with (from the 12-Evaluation folder):
#    python run_offline_eval.py v1
#    python run_offline_eval.py v2
#    python run_offline_eval.py v3
#
# Expected result: a 12-row table and a summary line. On our runs:
#   v1: correct_tools 0.83 | facts_present 0.83 | judge_reply 0.83
#   v2: correct_tools 1.00 | facts_present 1.00 | judge_reply 1.00
#   v3: correct_tools 1.00 | facts_present 0.92 | judge_reply 0.92
# The wording of the answers varies between runs; these scores did not.

import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))  # finds .env here or in a parent folder

from desk import build_desk, run_desk
from evaluators import correct_tools, facts_present, judge_reply

HERE = Path(__file__).parent
METRICS = ["correct_tools", "facts_present", "judge_reply"]


def load_golden(path=HERE / "golden_set.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate_example(desk, example: dict) -> dict:
    inputs = {"question": example["question"]}
    outputs = run_desk(desk, example["question"])
    scores = {
        "correct_tools": correct_tools(outputs, example),
        "facts_present": facts_present(outputs, example),
        "judge_reply": judge_reply(inputs, outputs, example),
    }
    return {"id": example["id"], **outputs, "scores": scores}


def main(version: str):
    desk = build_desk(version)
    rows = [evaluate_example(desk, ex) for ex in load_golden()]

    print(f"{'example':<16}{'tools':>7}{'facts':>7}{'judge':>7}")
    for row in rows:
        s = row["scores"]
        print(f"{row['id']:<16}" + "".join(f"{s[m]['score']:>7.2f}" for m in METRICS))

    averages = {m: sum(r["scores"][m]["score"] for r in rows) / len(rows) for m in METRICS}
    print(f"\n{version}: " + " | ".join(f"{m} {v:.2f}" for m, v in averages.items()))

    out = HERE / "results" / f"{version}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"version": version, "averages": averages,
                               "rows": rows}, indent=2))
    print(f"saved {out.relative_to(HERE)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v2")
