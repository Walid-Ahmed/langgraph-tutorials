# An LLM-as-judge metric in a few lines (Chapter 14, Section 14.6).
#
# A second model grades each answer against the expected one, using a written
# rubric, and returns its reason before its score. It reads meaning, so it can
# tell the reworded right answer from the similar-sounding wrong one, which the
# word-overlap heuristics in rule_and_heuristic.py could not.
#
# Prerequisites: OPENAI_API_KEY in a .env file (this folder or a parent folder).
# Run it with (from the repository root):
#    python 12-Evaluation/02-metric-types/llm_judge.py
# Expected result: a score from 0 to 1 and a one-sentence reason for each answer.
# Wording varies between runs.

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rule_and_heuristic import ANSWERS, REFERENCE

load_dotenv(find_dotenv(usecwd=True))

QUESTION = "When does flight EK924 leave Cairo?"
RUBRIC = """You grade an answer against an expected answer.
Check that the answer states the same facts; different wording is fine.
A wrong or missing fact lowers the grade.
Grade from 0 (wrong) to 10 (every expected fact, nothing contradicted)."""


# The judge's structured reply. The reason comes first, so the model explains before it scores.
class Grade(BaseModel):
    reason: str = Field(description="One sentence explaining the grade.")
    score: int = Field(ge=0, le=10)


judge = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Grade)


# LLM-as-judge: grade one answer against the expected answer, scaled from 0-10 to 0-1.
def correctness(answer: str) -> tuple[float, str]:
    grade = judge.invoke([
        ("system", RUBRIC),
        ("human", f"Question: {QUESTION}\nExpected answer: {REFERENCE}\nAnswer to grade: {answer}"),
    ])
    assert isinstance(grade, Grade)
    return grade.score / 10, grade.reason


if __name__ == "__main__":
    for name, answer in ANSWERS.items():
        score, reason = correctness(answer)
        print(f"{name}: {score:.1f} | {reason}")
