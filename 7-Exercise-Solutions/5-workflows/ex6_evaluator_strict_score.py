# Exercise 6 — Stricter evaluator with a numeric score
#
# Extends the Feedback schema with score: int (1-10).
# The router only accepts jokes scoring 8 or higher.
# Lower-scoring jokes are retried up to MAX_ITERATIONS times.
# Solution to Part 4 Exercise 3 in 05_evaluator_optimizer.md.

import sys
from pathlib import Path
from typing import TypedDict, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

MAX_ITERATIONS = 3
MIN_SCORE = 8


class State(TypedDict):
    topic: str
    joke: str
    feedback: str
    score: int
    iterations: int


class Feedback(BaseModel):
    score: int = Field(
        description="Rate the joke from 1 (not funny at all) to 10 (hilarious).",
    )
    feedback: str = Field(
        description="Explain the score and suggest how to improve the joke if the score is below 8.",
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
    grade = cast(Feedback, evaluator.invoke(f"Grade this joke: {state['joke']}"))
    return {
        "score": grade.score,
        "feedback": grade.feedback,
        "iterations": state.get("iterations", 0) + 1,
    }


def route_joke(state: State):
    if state["score"] >= MIN_SCORE:
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
print(f"\nFinal score: {state['score']}/10 — stopped after {state['iterations']} iteration(s)")
