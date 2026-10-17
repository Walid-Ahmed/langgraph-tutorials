# Run from the repository root, in two steps:
#   python Exercise-Solutions/12-evaluation/ex2_judge_agreement.py collect
#   (now grade every reply yourself in ex2_hand_labels.json, WITHOUT looking at
#    the judge's comments in LangSmith)
#   python Exercise-Solutions/12-evaluation/ex2_judge_agreement.py compare
#
# What it does:
#   Exercise 14.2. "collect" finds the latest desk-v1 experiment on the chapter's
#   "corniche-cycles-desk" dataset and saves its replies to ex2_v1_replies.json:
#   question, expected behavior and reply only, no scores. "compare" then:
#     1. reads the judge_reply feedback stored in that experiment (the chapter's
#        gpt-4o judge) and counts how often it agrees with your hand labels;
#     2. grades the SAME saved replies again with the chapter's rubric, REPEATS
#        times each, with gpt-4o and with gpt-4o-mini, and prints every verdict
#        next to yours, the agreement rate per judge, and the judge's reason
#        wherever it disagrees with you.
#
# Why it matters:
#   - An LLM judge is a second model, and the only way to know how much to trust
#     it is to measure it against a person. The agreement rate is that number.
#   - Both judges grade identical replies. The exercise's shortcut, re-running v1
#     with the judge switched, would also change the replies, and a change in
#     agreement could then come from the replies rather than from the judge.
#   - Grading each reply several times shows whether a verdict is stable. A judge
#     that agrees with you on one run and not the next isn't agreeing.
#   - The judges are rebuilt from the chapter's JUDGE_RUBRIC and Verdict, with only
#     the model name changed, so evaluators.py stays untouched. Nothing is written
#     back to LangSmith.
#
# Prerequisites:
#   pip install -r requirements.txt, then OPENAI_API_KEY and LANGSMITH_API_KEY in
#   the repo-root .env or the environment, and a desk-v1 experiment (run
#   `python run_experiment.py v1` in 12-Evaluation once).
#   compare: 72 judge calls (12 replies x 3 repeats x 2 models), a few US cents.
#
# Expected result (our runs, on experiment desk-v1-e18b3ce8):
#   stored gpt-4o verdicts in desk-v1-e18b3ce8: 12/12 agree
#   gpt-4o       agreement per run: 12/12, 12/12, 12/12  -> 100%
#   gpt-4o-mini  agreement per run: 11/12, 11/12, 12/12  -> 94%
# gpt-4o-mini's only disagreement is missing-type, which it fails on some runs
# and not others ("does not provide any pricing information"). A second
# "compare" gave gpt-4o-mini 12/12, 11/12, 12/12 (97%); gpt-4o stayed at 100%.
# Your replies, and therefore your numbers, will differ.

import json
import sys
from pathlib import Path

from dataset_copy import SOURCE

from run_experiment import client  # noqa: E402  (loads .env before the judge is used)
from evaluators import JUDGE_RUBRIC, Verdict  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

HERE = Path(__file__).parent
REPLIES = HERE / "ex2_v1_replies.json"
HAND_LABELS = HERE / "ex2_hand_labels.json"
JUDGE_MODELS = ["gpt-4o", "gpt-4o-mini"]
REPEATS = 3


def latest_v1_experiment():
    """The most recent desk-v1 experiment on the chapter's dataset that has runs in it."""
    experiments = sorted((p for p in client.list_projects(reference_dataset_name=SOURCE,
                                                          metadata={"version": "v1"})
                          if p.name.startswith("desk-v1-")),
                         key=lambda p: p.start_time, reverse=True)
    for experiment in experiments:
        runs = list(client.list_runs(project_name=experiment.name, is_root=True))
        if runs:
            return experiment.name, runs
    sys.exit("no desk-v1 experiment found: run `python run_experiment.py v1` in 12-Evaluation")


def collect():
    """Save the v1 replies with no scores, so the hand grading stays blind."""
    name, runs = latest_v1_experiment()
    examples = {ex.id: ex for ex in client.list_examples(dataset_name=SOURCE)}
    replies = []
    for r in runs:
        ex = examples[r.reference_example_id]
        replies.append({"id": ex.metadata["id"], "question": ex.inputs["question"],
                        "expected_behavior": ex.outputs["expected_behavior"],
                        "answer": r.outputs["answer"], "run_id": str(r.id)})
    replies.sort(key=lambda reply: reply["id"])
    REPLIES.write_text(json.dumps({"experiment": name, "replies": replies}, indent=2))
    for reply in replies:
        print(f"--- {reply['id']}\n{reply['answer']}\n")
    print(f"saved {len(replies)} replies from {name}; grade them in {HAND_LABELS.name}")


def grade(judge, reply: dict) -> Verdict:
    """The chapter's judge_reply prompt; returns the verdict and its reason."""
    return judge.invoke([
        ("system", JUDGE_RUBRIC),
        ("human",
         f"Customer question:\n{reply['question']}\n\n"
         f"Expected behavior:\n{reply['expected_behavior']}\n\n"
         f"Reply to grade:\n{reply['answer']}"),
    ])


def compare():
    saved = json.loads(REPLIES.read_text())
    replies = saved["replies"]
    mine = {k: v["label"] for k, v in json.loads(HAND_LABELS.read_text()).items()}

    # 1. The verdicts the chapter's experiment already stored, as judge_reply feedback.
    stored = {}
    run_ids = {reply["run_id"]: reply["id"] for reply in replies}
    for fb in client.list_feedback(run_ids=list(run_ids), feedback_key=["judge_reply"]):
        stored[run_ids[str(fb.run_id)]] = "pass" if fb.score == 1 else "fail"
    agree = sum(stored[i] == mine[i] for i in mine)
    print(f"stored gpt-4o verdicts in {saved['experiment']}: {agree}/{len(mine)} agree\n")

    # 2. Both judges on the same replies, REPEATS times each.
    graded = {}  # graded[model][run] maps example id -> Verdict
    for model in JUDGE_MODELS:
        judge = ChatOpenAI(model=model, temperature=0).with_structured_output(Verdict)
        graded[model] = [{r["id"]: grade(judge, r) for r in replies} for _ in range(REPEATS)]

    print(f"{'example':<16}{'me':<6}" + "".join(f"{m:<20}" for m in JUDGE_MODELS))
    for i in sorted(mine):
        cells = ["/".join(run[i].verdict for run in graded[m]) for m in JUDGE_MODELS]
        agreed = all(run[i].verdict == mine[i] for m in JUDGE_MODELS for run in graded[m])
        print(f"{i:<16}{mine[i]:<6}" + "".join(f"{c:<20}" for c in cells)
              + ("" if agreed else "  <- disagreement"))

    print()
    for model in JUDGE_MODELS:
        per_run = [sum(run[i].verdict == mine[i] for i in mine) for run in graded[model]]
        rate = sum(per_run) / (len(mine) * REPEATS)
        print(f"{model:<12} agreement per run: "
              + ", ".join(f"{n}/{len(mine)}" for n in per_run) + f"  -> {rate:.0%}")

    # A wrong grade is easier to judge from its reason: print one per disagreement.
    print()
    for model in JUDGE_MODELS:
        for i in sorted(mine):
            wrong = [run[i] for run in graded[model] if run[i].verdict != mine[i]]
            if wrong:
                print(f"{model} on {i} ({len(wrong)}/{REPEATS} runs said {wrong[0].verdict}): "
                      f"{wrong[0].reason}")


if __name__ == "__main__":
    {"collect": collect, "compare": compare}[sys.argv[1] if len(sys.argv) > 1 else "compare"]()
