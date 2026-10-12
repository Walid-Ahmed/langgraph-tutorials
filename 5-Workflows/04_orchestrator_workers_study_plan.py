# Run from the repository root:
#   python "5-Workflows/04_orchestrator_workers_study_plan.py"
#
# What it does:
#   Turns a learning goal into a weekly study plan. A planner node asks the
#   model to split the goal into weeks (the model decides how many); a
#   dispatcher sends one Send per week to a write_week worker; the workers run
#   in parallel, each appending its week to a shared list; assemble_plan sorts
#   the weeks and joins them.
#
# What it demonstrates (Chapter 5 — orchestrator–workers):
#   - the number of workers is decided at run time, by the planner, via Send
#   - each worker gets its own small input (one week), not the whole state
#   - the shared result field needs a reducer (operator.add) because every
#     worker writes to it in the same step
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (the number of weeks varies, usually 3-6):
#   Planner: 4 week(s)
#   ## Week 1: ...
#   ## Week 2: ...

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class Week(BaseModel):
    number: int = Field(description="1 for the first week, 2 for the second, ...")
    focus: str = Field(description="what the learner studies that week, in a few words")


class Plan(BaseModel):
    weeks: list[Week] = Field(description="between 3 and 6 weeks, in order")


planner = llm.with_structured_output(Plan)


class State(TypedDict):
    goal: str
    weeks: list[Week]
    week_plans: Annotated[list[tuple[int, str]], operator.add]   # every worker appends here
    study_plan: str


class WeekTask(TypedDict):          # what one worker receives
    goal: str
    week: Week


def plan_weeks(state: State) -> dict:
    plan = cast(Plan, planner.invoke(f"Split this learning goal into weekly steps: {state['goal']}"))
    print(f"Planner: {len(plan.weeks)} week(s)")
    return {"weeks": plan.weeks}


def dispatch_weeks(state: State) -> list[Send]:
    """One Send per planned week: the count is only known now, at run time."""
    return [Send("write_week", {"goal": state["goal"], "week": week}) for week in state["weeks"]]


def write_week(state: WeekTask) -> dict:
    """A worker sees only its own WeekTask, not the whole graph state."""
    week = state["week"]
    text = llm.invoke(f"Learning goal: {state['goal']}\nWrite the plan for week {week.number}, "
                      f"focused on {week.focus}: 3 bullet points of what to do, and one small "
                      f"exercise. Start with the heading '## Week {week.number}: {week.focus}'.").text
    return {"week_plans": [(week.number, text)]}


def assemble_plan(state: State) -> dict:
    """Workers finish in any order, so sort by week number before joining."""
    ordered = [text for _, text in sorted(state["week_plans"])]
    return {"study_plan": "\n\n".join(ordered)}


builder = StateGraph(State)
builder.add_node("plan_weeks", plan_weeks)
builder.add_node("write_week", write_week)
builder.add_node("assemble_plan", assemble_plan)
builder.add_edge(START, "plan_weeks")
builder.add_conditional_edges("plan_weeks", dispatch_weeks, ["write_week"])
builder.add_edge("write_week", "assemble_plan")
builder.add_edge("assemble_plan", END)
graph = builder.compile()


def main() -> None:
    plot_graph(graph, str(Path(__file__).resolve().parent / "diagrams" / "04_orchestrator_workers_study_plan_graph.png"),
               print_mermaid=False)
    result = graph.invoke({"goal": "Learn enough SQL to analyze my team's sales data"})  # type: ignore[arg-type]
    print("\n" + result["study_plan"])


if __name__ == "__main__":
    main()
