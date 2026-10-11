# Step 13 in 6-Agents: a supervisor agent decides which specialist works next.
#
# This script answers a cooking question with a small team. A supervisor reads
# the request, sends it to a nutrition agent and/or a cost agent, checks each
# report, sends a weak report back with feedback, and finally hands everything
# to a writer that produces a recipe card and saves it to a file.
#
#                 +--> nutrition --+
#     START -> supervisor <--------+        (every specialist reports back)
#                 +--> cost -------+
#                 +--> writer -> END
#
# Why this matters:
# - In 11_parallel_subgraphs.py the EDGES decided that both agents run. Here a
#   MODEL decides, one hop at a time, after reading each report. A
#   calories-only question never pays for the cost agent.
# - The supervisor's choice is a structured Decision (a Literal), not free
#   text, so a typo in the reply can't send the run down the wrong branch.
# - The supervisor returns a Command: one return updates state and picks the
#   next node, so there is no router function and no add_conditional_edges.
# - The loop is capped in code (MAX_WORKER_RUNS). The prompt asks the
#   supervisor to finish; the counter makes sure it does.
# - It does NOT search the web or call a food API. Calories and prices are
#   sample numbers in PANTRY below, so the arithmetic is easy to check by hand.
#
# Before you run this:
# 1. Put OPENAI_API_KEY=... in the repo-root .env
# 2. pip install -r requirements.txt
#
# Run it from the repository root with:
#    python "6-Agents/13_supervisor.py"
#
# It prints each hop, then the recipe card, and saves the card to
# 6-Agents/logs/recipe_card.txt (overwritten on every run, ignored by Git).

import os
import sys
from operator import add
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from util import plot_graph  # noqa: E402

MODEL = "openai:gpt-4o-mini"
MAX_WORKER_RUNS = 4          # two specialists, one retry each, then we stop
RECIPE_CARD_FILE = Path(__file__).resolve().parent / "logs" / "recipe_card.txt"

REQUESTS = [
    ("I'm making lentil soup for 4 people: 300 g red lentils, 200 g carrots, "
     "150 g onion and 30 g olive oil. How many calories and what cost per serving?"),
    "How many calories per serving in 300 g red lentils and 30 g olive oil, for 2 people?",
]

# Sample data, so the example needs no food or grocery API.
PANTRY = {
    "red lentils": {"kcal_per_100g": 358, "cad_per_kg": 4.50},
    "carrots":     {"kcal_per_100g": 41,  "cad_per_kg": 2.20},
    "onion":       {"kcal_per_100g": 40,  "cad_per_kg": 2.00},
    "olive oil":   {"kcal_per_100g": 884, "cad_per_kg": 14.00},
}


def _unknown_ingredient(ingredient: str) -> str:
    # Returned as text, not raised: the model can read it and correct itself.
    return f"Unknown ingredient {ingredient!r}. Known ingredients: {', '.join(PANTRY)}."


# ---------------------------------------------------------
# 1. The tools: each specialist gets only the ones for its part
# ---------------------------------------------------------
@tool
def get_calories(ingredient: str, grams: float) -> str:
    """Get the calories (kcal) in a given weight of one ingredient."""
    item = PANTRY.get(ingredient.strip().lower())
    if item is None:
        return _unknown_ingredient(ingredient)
    return f"{grams:g} g {ingredient}: {item['kcal_per_100g'] * grams / 100:.1f} kcal"


@tool
def get_price(ingredient: str, grams: float) -> str:
    """Get the price in CAD of a given weight of one ingredient."""
    item = PANTRY.get(ingredient.strip().lower())
    if item is None:
        return _unknown_ingredient(ingredient)
    return f"{grams:g} g {ingredient}: {item['cad_per_kg'] * grams / 1000:.2f} CAD"


@tool
def add_numbers(amounts: list[float]) -> float:
    """Add a list of numbers and return the total. Use this instead of mental arithmetic."""
    return round(sum(amounts), 2)


@tool
def per_serving(total: float, servings: int) -> str:
    """Divide a total by the number of servings."""
    if servings <= 0:
        return "servings must be a positive whole number."
    return f"{total / servings:.2f}"


# ---------------------------------------------------------
# 2. The specialists: two create_agent graphs, as in 11_parallel_subgraphs.py
# ---------------------------------------------------------
nutrition_agent = create_agent(
    model=MODEL,
    tools=[get_calories, add_numbers, per_serving],
    system_prompt=("You are a nutrition specialist. Report only calories: look up each "
                   "ingredient, add them with add_numbers, and give the total and the "
                   "calories per serving in at most three sentences. Never guess a number."),
)

cost_agent = create_agent(
    model=MODEL,
    tools=[get_price, add_numbers, per_serving],
    system_prompt=("You are a grocery cost specialist. Report only cost: look up each "
                   "ingredient, add the prices with add_numbers, and give the total and the "
                   "cost per serving in CAD in at most three sentences. Never guess a number."),
)


# ---------------------------------------------------------
# 3. The supervisor's decision, as structured output
# ---------------------------------------------------------
class Decision(BaseModel):
    """The supervisor's choice of what happens next."""
    next: Literal["nutrition", "cost", "finish"] = Field(
        description="The specialist to run next, or 'finish' when every report "
                    "the request needs is acceptable.")
    feedback: str = Field(
        default="",
        description="When sending a specialist back to redo its report, what to "
                    "fix. Empty on a specialist's first run.")


# The reply is parsed into a Decision, so `next` can only be one of the three
# values above.
supervisor_llm = init_chat_model(MODEL, temperature=0).with_structured_output(Decision)
writer_llm = init_chat_model(MODEL, temperature=0)    # no tools: it only writes

SUPERVISOR_PROMPT = """You supervise two specialists for cooking questions.

- nutrition: reports total calories and calories per serving
- cost: reports total cost and cost per serving

Decide what happens next:
1. Run only the specialists the request needs.
2. A report is acceptable when it answers its part of the request and gives a
   per-serving number.
3. If a report is missing or wrong, send that specialist back with feedback
   that says exactly what to fix.
4. When every needed report is acceptable, choose finish.

Request:
{request}

Nutrition report:
{nutrition_report}

Cost report:
{cost_report}
"""


# ---------------------------------------------------------
# 4. The parent state: no `messages` key, one writer per report field
# ---------------------------------------------------------
class KitchenState(TypedDict, total=False):
    request: str             # set by the caller; everyone reads it
    nutrition_report: str    # written only by the nutrition node
    cost_report: str         # written only by the cost node
    feedback: str            # written by the supervisor for the next specialist
    worker_runs: int         # how many specialist runs so far (the loop cap)
    path: Annotated[list[str], add]    # the supervisor's choices, in order
    final_answer: str        # written only by the writer node


# ---------------------------------------------------------
# 5. The nodes
# ---------------------------------------------------------
def supervisor_node(state: KitchenState) -> Command[Literal["nutrition", "cost", "writer"]]:
    """Check the reports so far and choose the next hop."""
    worker_runs = state.get("worker_runs", 0)
    if worker_runs >= MAX_WORKER_RUNS:
        # Code, not the prompt, guarantees the loop ends.
        return Command(goto="writer", update={"path": ["writer (run limit reached)"]})

    decision = supervisor_llm.invoke(SUPERVISOR_PROMPT.format(
        request=state["request"],
        nutrition_report=state.get("nutrition_report", "(none yet)"),
        cost_report=state.get("cost_report", "(none yet)"),
    ))
    if decision.next == "finish":
        return Command(goto="writer", update={"path": ["writer"]})

    # feedback is overwritten on every hop, so a specialist never receives
    # feedback that was meant for the other one.
    return Command(
        goto=decision.next,
        update={"feedback": decision.feedback,
                "worker_runs": worker_runs + 1,
                "path": [decision.next]},
    )


def run_specialist(agent, state: KitchenState, report_field: str) -> dict:
    """Run one specialist and return its final answer as one state field."""
    task = f"Request:\n{state['request']}"
    previous_report = state.get(report_field)
    if previous_report and state.get("feedback"):
        # A retry: show the specialist its own report and what to fix.
        task += (f"\n\nYour previous report:\n{previous_report}"
                 f"\n\nThe supervisor asked you to fix this:\n{state['feedback']}")
    result = agent.invoke({"messages": [HumanMessage(task)]})
    # Only the conclusion crosses back; the tool calls stay inside the agent.
    return {report_field: result["messages"][-1].content}


def nutrition_node(state: KitchenState) -> dict:
    return run_specialist(nutrition_agent, state, "nutrition_report")


def cost_node(state: KitchenState) -> dict:
    return run_specialist(cost_agent, state, "cost_report")


def writer_node(state: KitchenState) -> dict:
    """Merge the reports into a recipe card and save it."""
    prompt = (f"Request:\n{state['request']}\n\n"
              f"Nutrition report:\n{state.get('nutrition_report', '(not requested)')}\n\n"
              f"Cost report:\n{state.get('cost_report', '(not requested)')}\n\n"
              "Write a short recipe card that answers the request, using only "
              "the reports above. Do not mention the specialists.")
    final_answer = writer_llm.invoke(prompt).content
    # The file name is fixed in code; the model only supplies the text.
    RECIPE_CARD_FILE.parent.mkdir(exist_ok=True)
    RECIPE_CARD_FILE.write_text(final_answer, encoding="utf-8")
    return {"final_answer": final_answer}


# ---------------------------------------------------------
# 6. The wiring: a hub and its spokes
# ---------------------------------------------------------
builder = StateGraph(KitchenState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("nutrition", nutrition_node)
builder.add_node("cost", cost_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "supervisor")
builder.add_edge("nutrition", "supervisor")    # every specialist reports back
builder.add_edge("cost", "supervisor")
builder.add_edge("writer", END)
# No edge out of supervisor: its Command decides (nutrition, cost, or writer).
graph = builder.compile()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repo-root .env.")

    png = Path(__file__).resolve().parent / "diagrams" / "13_supervisor.png"
    png.parent.mkdir(exist_ok=True)
    try:
        plot_graph(graph, str(png), print_mermaid=False)
    except Exception as error:    # the PNG needs network access; the run does not
        print(f"(graph picture skipped: {type(error).__name__})")

    for request in REQUESTS:
        print(f"\nRequest: {request}\n")
        route = ["supervisor"]
        # stream_mode="updates" shows which node wrote which field, step by step.
        for update in graph.stream({"request": request}, stream_mode="updates"):
            for node_name, written in update.items():
                if node_name == "supervisor":
                    route.append(written["path"][-1])
                    feedback = written.get("feedback")
                    print(f"[supervisor] -> {route[-1]}"
                          + (f"   feedback: {feedback}" if feedback else ""))
                else:
                    for field, value in written.items():
                        print(f"[{node_name}] wrote {field}:\n{value}\n")

        # Expected shape, request 1: nutrition and cost each run (either order),
        # then writer. Request 2: nutrition only, then writer. A specialist
        # appears twice when the supervisor sent it back. Wording varies.
        print("Route:", " -> ".join(route))
        if route[-1].startswith("writer"):    # no card unless the writer ran
            print(f"Recipe card saved to {RECIPE_CARD_FILE}")


if __name__ == "__main__":
    main()
