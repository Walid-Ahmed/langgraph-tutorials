# Exercise 4 — Add a loop guard to the evaluator-optimizer
#
# Adds an iterations counter to State. The loop stops after
# MAX_ITERATIONS rejections regardless of the evaluator verdict.
# Solution to Part 4 Exercise 1 in 05_evaluator_optimizer.md.

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
    joke: str
    feedback: str
    funny_or_not: str
    iterations: int  # incremented by evaluator on every pass


class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )


evaluator = llm.with_structured_output(Feedback)


def llm_call_generator(state: State):
    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def llm_call_evaluator(state: State):
    grade = cast(Feedback, evaluator.invoke(f"Grade the joke: {state['joke']}"))
    return {
        "funny_or_not": grade.grade,
        "feedback": grade.feedback,
        "iterations": state.get("iterations", 0) + 1,
    }


def route_joke(state: State):
    if state["funny_or_not"] == "funny":
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
    route_joke,
    {
        "Accepted": END,
        "Max iterations reached": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

workflow = builder.compile()
plot_graph(workflow)

state = workflow.invoke({"topic": "Cats", "iterations": 0})  # type: ignore[arg-type]
print(state["joke"])
print(f"\nStopped after {state['iterations']} iteration(s) — verdict: {state['funny_or_not']}")
