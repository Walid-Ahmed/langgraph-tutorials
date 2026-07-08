# Evaluator-optimizer variant: same generate -> evaluate -> retry loop as
# 05_evaluator_optimizer.py, but adds an iteration counter so the router
# force-exits after MAX_ITERATIONS instead of retrying forever.

import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

MAX_ITERATIONS = 3


# ---------------------------------------------------------
# 1. State
#
# Same as the base example with one addition:
# - iterations: incremented each time the joke is rejected
#               so the router can enforce a hard stop
# ---------------------------------------------------------

class State(TypedDict):
    topic: str
    joke: str
    feedback: str
    funny_or_not: str
    iterations: int


# ---------------------------------------------------------
# 2. Structured output schema (unchanged)
# ---------------------------------------------------------

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )


evaluator = llm.with_structured_output(Feedback)


# ---------------------------------------------------------
# 3. Generator node (unchanged)
# ---------------------------------------------------------

def llm_call_generator(state: State):
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


# ---------------------------------------------------------
# 4. Evaluator node
#
# Same grading logic, but also increments iterations so the
# router always has an up-to-date count regardless of verdict.
# ---------------------------------------------------------

def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    grade = cast(Feedback, evaluator.invoke(f"Grade the joke: {state['joke']}"))
    return {
        "funny_or_not": grade.grade,
        "feedback": grade.feedback,
        "iterations": state.get("iterations", 0) + 1,
    }


# ---------------------------------------------------------
# 5. Conditional edge (router)
#
# Three exit conditions:
#   - joke is funny              → Accepted (end normally)
#   - limit reached              → Max iterations reached (end early)
#   - joke is not funny yet      → Rejected + Feedback (retry)
# ---------------------------------------------------------

def route_joke(state: State):
    """Route back to joke generator or end based on verdict and iteration count"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    if state["iterations"] >= MAX_ITERATIONS:
        return "Max iterations reached"
    return "Rejected + Feedback"


# ---------------------------------------------------------
# 6. Build graph
# ---------------------------------------------------------

optimizer_builder = StateGraph(State)

optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {
        "Accepted": END,
        "Max iterations reached": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

optimizer_workflow = optimizer_builder.compile()

plot_graph(optimizer_workflow)

# ---------------------------------------------------------
# 7. Run
# ---------------------------------------------------------

state = optimizer_workflow.invoke({"topic": "Cats", "iterations": 0})  # type: ignore[arg-type]

print(state["joke"])
print(f"\nFinished after {state['iterations']} iteration(s) — verdict: {state['funny_or_not']}")
