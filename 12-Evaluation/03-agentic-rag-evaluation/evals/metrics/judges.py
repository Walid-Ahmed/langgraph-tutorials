# LLM-as-judge metrics: a second model grades the answer against a rubric.
#
# Each judge has a fixed rubric (the evaluation steps), reads only the fields it
# needs, and returns its reason before its score: structured output writes fields
# in schema order, so the model explains before it commits. The 0-10 grade is
# scaled to 0-1 so it sits beside the other metrics.
#
# Prerequisites: OPENAI_API_KEY. The judge model is created on first use, so
# importing this module costs nothing.

from functools import cache

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from evals.metrics.result import MetricResult
from src import config

CORRECTNESS_STEPS = """You grade answers from a company's internal knowledge-base chatbot.
1. Check whether the answer states the same facts as the expected answer.
   Different wording and number formats ($500K, $500,000, 500000) are fine.
2. Reject unsupported or contradictory claims; a missing key fact lowers the grade.
Grade from 0 (wrong) to 10 (every expected fact present, nothing contradicted)."""

RELEVANCE_STEPS = """You grade answers from a company's internal knowledge-base chatbot.
1. Check whether the answer addresses the question directly.
2. Penalize unrelated or tangential information.
Grade from 0 (off-topic) to 10 (answers exactly what was asked)."""


class Grade(BaseModel):
    reason: str = Field(description="One or two sentences explaining the grade.")
    score: int = Field(ge=0, le=10)


@cache
def _judge():
    return ChatOpenAI(model=config.JUDGE_MODEL, temperature=0).with_structured_output(Grade)


def _grade(name: str, steps: str, prompt: str) -> MetricResult:
    grade = _judge().invoke([("system", steps), ("human", prompt)])
    assert isinstance(grade, Grade)
    return MetricResult(name, grade.score / 10, grade.reason)


def correctness_judge(question: str, answer: str, expected_answer: str) -> MetricResult:
    """Reference-based: does the answer match the golden expected answer?"""
    return _grade("correctness", CORRECTNESS_STEPS,
                  f"Question:\n{question}\n\nExpected answer:\n{expected_answer}\n\nAnswer to grade:\n{answer}")


def relevance_judge(question: str, answer: str) -> MetricResult:
    """Reference-free: does the answer address the question? Usable offline and online."""
    return _grade("relevance", RELEVANCE_STEPS, f"Question:\n{question}\n\nAnswer to grade:\n{answer}")
