# Exercise 5 — Swap the domain: headline writer with clear/unclear evaluator
#
# Replaces the joke generator with a headline writer.
# The Feedback schema grades on "clear" / "unclear" instead of
# "funny" / "not funny". The loop retries until the headline
# is graded clear or MAX_ITERATIONS is reached.
# Solution to Part 4 Exercise 2 in 05_evaluator_optimizer.md.

import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

MAX_ITERATIONS = 3


class State(TypedDict):
    topic: str
    headline: str
    feedback: str
    clear_or_not: str
    iterations: int


class Feedback(BaseModel):
    grade: Literal["clear", "unclear"] = Field(
        description="Decide if the headline is clear and compelling or unclear.",
    )
    feedback: str = Field(
        description="If the headline is unclear, provide feedback on how to improve it.",
    )


evaluator = llm.with_structured_output(Feedback)


def llm_call_generator(state: State):
    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a news headline about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a concise, compelling news headline about {state['topic']}")
    return {"headline": msg.content}


def llm_call_evaluator(state: State):
    grade = cast(Feedback, evaluator.invoke(f"Grade this headline: {state['headline']}"))
    return {
        "clear_or_not": grade.grade,
        "feedback": grade.feedback,
        "iterations": state.get("iterations", 0) + 1,
    }


def route_headline(state: State):
    if state["clear_or_not"] == "clear":
        return "Accepted"
    if state["iterations"] >= MAX_ITERATIONS:
        return "Max iterations reached"
    return "Rejected + Feedback"


builder = StateGraph(State)
builder.add_node("llm_call_generator", llm_call_generator)
builder.add_node("llm_call_evaluator", llm_call_evaluator)
builder.add_edge(START, "llm_call_generator")
builder.add_edge("llm_call_generator", "llm_call_evaluator")
builder.add_conditional_edges(
    "llm_call_evaluator",
    route_headline,
    {
        "Accepted": END,
        "Max iterations reached": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

workflow = builder.compile()
plot_graph(workflow)

state = workflow.invoke({"topic": "Climate change", "iterations": 0})  # type: ignore[arg-type]
print(state["headline"])
print(f"\nStopped after {state['iterations']} iteration(s) — verdict: {state['clear_or_not']}")
