# Three evaluators for the Corniche Cycles help desk: route, facts, and a judge.
#
# Each evaluator takes what the agent produced (`outputs`: answer + tools_called)
# and what the golden example says should happen (`reference`), and returns a
# score from 0 to 1 plus a short reason. They are imported by run_offline_eval.py
# and wrapped for LangSmith by langsmith_experiment.py; not meant to be run alone.
#
# Why this matters:
# - correct_tools grades the TRAJECTORY: did the agent take the right route?
#   It is code, free and deterministic, and compares tool names as a multiset,
#   so order doesn't matter but a missing or extra call does.
# - facts_present grades the RESULT with code too: every must_include string
#   has to appear in the answer. It can't judge tone, only presence.
# - judge_reply is an LLM-AS-JUDGE: a second model reads the answer against the
#   example's expected_behavior and a fixed rubric. It is the only one that can
#   grade "asked for the missing bike type politely", and the only one that
#   costs money and can itself be wrong.
#
# Before you use this:
# 1. OPENAI_API_KEY in the environment (judge_reply only; the others are offline).

from collections import Counter
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


def correct_tools(outputs: dict, reference: dict) -> dict:
    """1 if the agent called exactly the expected tools (any order), else 0."""
    got = Counter(outputs["tools_called"])
    want = Counter(reference["expected_tools"])
    if got == want:
        return {"score": 1, "reason": "route as expected"}
    missing = list((want - got).elements())
    extra = list((got - want).elements())
    return {"score": 0, "reason": f"missing {missing}, extra {extra}"}


def facts_present(outputs: dict, reference: dict) -> dict:
    """Share of the must_include strings found in the answer (1.0 when none are required)."""
    required = reference["must_include"]
    if not required:
        return {"score": 1, "reason": "nothing required"}
    answer = outputs["answer"].replace(",", "")  # "EGP 1,200" should match "1200"
    missing = [fact for fact in required if fact not in answer]
    score = (len(required) - len(missing)) / len(required)
    return {"score": score, "reason": f"missing {missing}" if missing else "all present"}


JUDGE_RUBRIC = """You grade replies from the help desk of a bike-rental shop.
A reply PASSES only if all of these hold:
1. It does what the expected behavior describes.
2. Every price or bike count it states matches the expected behavior.
3. It does not promise refunds, discounts or anything the expected behavior doesn't allow.
   Saying that a request will be passed to a staff member is not a promise.
4. It is polite and reasonably brief.
Grade the reply, not the customer. Be strict about facts and lenient about wording."""


class Verdict(BaseModel):
    reason: str = Field(description="One sentence explaining the grade.")
    verdict: Literal["pass", "fail"]


# The judge is a stronger model than the desk it grades (gpt-4o vs gpt-4o-mini).
judge = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Verdict)


def judge_reply(inputs: dict, outputs: dict, reference: dict) -> dict:
    """LLM-as-judge: 1 if a second model says the reply meets the rubric, else 0."""
    result = judge.invoke([
        ("system", JUDGE_RUBRIC),
        ("human",
         f"Customer question:\n{inputs['question']}\n\n"
         f"Expected behavior:\n{reference['expected_behavior']}\n\n"
         f"Reply to grade:\n{outputs['answer']}"),
    ])
    return {"score": int(result.verdict == "pass"), "reason": result.reason}
