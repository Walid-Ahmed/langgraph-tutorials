# Evaluators for the Corniche Cycles help desk, written in LangSmith's shape.
#
# Every function here is a LangSmith evaluator: LangSmith calls it once per
# example and passes, BY ARGUMENT NAME, any of `inputs` (the example's inputs),
# `outputs` (what the target returned) and `reference_outputs` (the example's
# expectations). Each returns {"score": ..., "comment": ...}, and LangSmith stores
# that as feedback named after the function. Imported by run_experiment.py,
# component_eval.py and regression_gate.py; not meant to be run alone.
#
# Why this matters:
# - correct_tools grades the end-to-end TRAJECTORY: did the whole run call the
#   right tools, as many times as expected, in any order? Code, free, repeatable.
# - first_step_calls grades ONE COMPONENT: did the `llm` node ask for the right
#   tools with the right ARGUMENTS? It catches "kids" vs "city", or 4 hours vs 5,
#   before any tool runs.
# - facts_present grades the RESULT with code: every must_include string has to
#   appear in the answer.
# - judge_reply is an LLM-AS-JUDGE: a second model grades the answer against the
#   example's expected_behavior and a fixed rubric. It's the only one that can
#   grade "asked for the missing bike type politely", and the only one that
#   costs money and can itself be wrong.
#
# Before you use this:
# 1. OPENAI_API_KEY in the environment (judge_reply only; the others are offline).

from collections import Counter
from functools import cache
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


def _calls(calls: list[dict], with_args: bool) -> Counter:
    """A multiset of tool calls, optionally with their arguments (strings compared without case)."""
    def key(call):
        if not with_args:
            return call["name"]
        args = {k: v.lower() if isinstance(v, str) else v for k, v in call["args"].items()}
        return call["name"], tuple(sorted(args.items()))
    return Counter(key(c) for c in calls)


def correct_tools(outputs: dict, reference_outputs: dict) -> dict:
    """End to end: 1 if the run called exactly the expected tools (any order), else 0."""
    got = Counter(outputs["tools_called"])
    want = _calls(reference_outputs["expected_calls"], with_args=False)
    if got == want:
        return {"score": 1, "comment": "route as expected"}
    return {"score": 0, "comment": f"missing {list((want - got).elements())}, "
                                   f"extra {list((got - want).elements())}"}


def first_step_calls(outputs: dict, reference_outputs: dict) -> dict:
    """Component: 1 if the llm node's first reply asks for the expected calls, arguments included."""
    got = _calls(outputs["tool_calls"], with_args=True)
    want = _calls(reference_outputs["expected_calls"], with_args=True)
    if got == want:
        return {"score": 1, "comment": "calls and arguments as expected"}
    def show(calls):
        return ", ".join(sorted(f"{n}({', '.join(f'{k}={v}' for k, v in a)})" for n, a in calls.elements())) or "no call"
    return {"score": 0, "comment": f"expected {show(want)}; got {show(got)}"}


def facts_present(outputs: dict, reference_outputs: dict) -> dict:
    """Share of the must_include strings found in the answer (1.0 when none are required)."""
    required = reference_outputs["must_include"]
    if not required:
        return {"score": 1, "comment": "nothing required"}
    answer = outputs["answer"].replace(",", "")  # "EGP 1,200" should match "1200"
    missing = [fact for fact in required if fact not in answer]
    score = (len(required) - len(missing)) / len(required)
    return {"score": score, "comment": f"missing {missing}" if missing else "all present"}


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


@cache
def get_judge():
    """Create the judge only when used, so code-only checks need no API key."""
    return ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Verdict)


def judge_reply(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge: 1 if a second model says the reply meets the rubric, else 0."""
    result = get_judge().invoke([
        ("system", JUDGE_RUBRIC),
        ("human",
         f"Customer question:\n{inputs['question']}\n\n"
         f"Expected behavior:\n{reference_outputs['expected_behavior']}\n\n"
         f"Reply to grade:\n{outputs['answer']}"),
    ])
    return {"score": int(result.verdict == "pass"), "comment": result.reason}
