# Evaluator-optimizer workflow: a generator writes a joke, an evaluator LLM
# grades it via structured output, and a router loops back to the generator
# with feedback until the joke is judged "funny" — an unbounded retry loop.

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


# ---------------------------------------------------------
# 1. State
#
# Shared memory passed between nodes on every graph step.
# - topic       : set once at invocation, never changes
# - joke        : written by the generator, read by the evaluator
# - funny_or_not: written by the evaluator, read by the router
# - feedback    : written by the evaluator, fed back into the generator
#                 when the joke is rejected so it can improve
# ---------------------------------------------------------

class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str


# ---------------------------------------------------------
# 2. Structured output schema for the evaluator
#
# with_structured_output forces the LLM to return a Feedback
# object rather than free text, giving the router a reliable
# "funny" / "not funny" signal to branch on.
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
# 3. Generator node
#
# On the first pass there is no feedback, so a plain prompt
# is used.  On every retry the evaluator's feedback is
# injected so the generator can incorporate the critique.
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
# Grades the current joke and always returns both a verdict
# and improvement feedback.  The structured output schema
# guarantees the grade is one of the two expected literals.
# ---------------------------------------------------------

def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    # Cast needed because with_structured_output returns Runnable[..., dict]
    # at the type level even though it actually returns a Feedback instance.
    grade = cast(Feedback, evaluator.invoke(f"Grade the joke: {state['joke']}"))
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}


# ---------------------------------------------------------
# 5. Conditional edge (router)
#
# The router reads the evaluator verdict and picks the next
# node.  Returning "Accepted" exits the loop; returning
# "Rejected + Feedback" sends execution back to the generator
# with the evaluator's notes still in state.
# ---------------------------------------------------------

def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


# ---------------------------------------------------------
# 6. Build graph
#
# Linear spine: generator -> evaluator
# Conditional back-edge: evaluator -> generator (on rejection)
#                        evaluator -> END        (on acceptance)
# ---------------------------------------------------------

optimizer_builder = StateGraph(State)

optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # label returned by route_joke : next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

optimizer_workflow = optimizer_builder.compile()

plot_graph(optimizer_workflow)

# ---------------------------------------------------------
# 7. Run
# ---------------------------------------------------------

state = optimizer_workflow.invoke({"topic": "Cats"})  # type: ignore[arg-type]
print(state["joke"])
