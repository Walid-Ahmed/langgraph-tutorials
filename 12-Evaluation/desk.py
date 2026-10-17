# Corniche Cycles help desk - the agent that every script in this folder evaluates.
#
# This module builds a small tool-calling agent for a fictional bike-rental shop
# in Alexandria. It has two tools (a price quote and a station stock check) and
# the Chapter 7 loop: llm -> tools -> llm, until the model stops asking for tools.
# It exports build_desk(version) and run_desk(desk, question); it is not meant to
# be run on its own.
#
# Why this matters:
# - An evaluation needs a system under test whose correct answers you KNOW. Both
#   tools read fixed tables, so "an electric bike for 3 hours costs EGP 225" is a
#   fact the golden dataset can check, not an opinion.
# - The desk comes in three versions, each a (system prompt, tools) pair:
#     v1  the first draft
#     v2  v1 with the clarification rule fixed after the first offline run
#     v3  v2 plus a "harmless" refactor of the pricing tool
#   compare_runs.py shows what the v3 refactor broke.
# - run_desk() returns the final answer, the names of the tools called and what
#   those tools returned, so an evaluator can grade the result, the route the
#   agent took to it, and whether the answer sticks to the evidence.
#
# Before you use this:
# 1. pip install -r requirements.txt   (from the repository root)
# 2. OPENAI_API_KEY in a .env file in this folder or the repository root.
#
# Imported by run_offline_eval.py, langsmith_experiment.py and online_feedback.py.

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

BikeType = Literal["city", "electric", "kids"]

HOURLY_RATE = {"city": 40, "electric": 75, "kids": 25}  # EGP per hour
LONG_RENTAL_HOURS = 5                                   # 5 hours or more...
LONG_RENTAL_DISCOUNT = 0.20                             # ...gets 20% off

STOCK = {
    "Stanley Bridge": {"city": 6, "electric": 2, "kids": 3},
    "Montaza Gate": {"city": 4, "electric": 0, "kids": 5},
    "Bibliotheca": {"city": 9, "electric": 4, "kids": 0},
}


@tool
def rental_quote(bike_type: BikeType, hours: int) -> str:
    """Price a rental in Egyptian pounds (EGP). Applies the long-rental discount."""
    base = HOURLY_RATE[bike_type] * hours
    if hours >= LONG_RENTAL_HOURS:
        total = round(base * (1 - LONG_RENTAL_DISCOUNT))
        return f"{bike_type} bike, {hours} h: EGP {total} (EGP {base} minus 20% long-rental discount)"
    return f"{bike_type} bike, {hours} h: EGP {base}"


def discount_for(hours: int) -> float:
    """The v3 refactor moved the discount rule into a helper and changed >= to >."""
    return LONG_RENTAL_DISCOUNT if hours > LONG_RENTAL_HOURS else 0.0


@tool("rental_quote")  # same tool name, so the model and the golden set see no change
def rental_quote_refactored(bike_type: BikeType, hours: int) -> str:
    """Price a rental in Egyptian pounds (EGP). Applies the long-rental discount."""
    base = HOURLY_RATE[bike_type] * hours
    discount = discount_for(hours)
    if discount:
        total = round(base * (1 - discount))
        return f"{bike_type} bike, {hours} h: EGP {total} (EGP {base} minus 20% long-rental discount)"
    return f"{bike_type} bike, {hours} h: EGP {base}"


@tool
def bikes_available(station: str) -> str:
    """How many bikes of each type are free right now at one station."""
    for name, counts in STOCK.items():
        if station.strip().lower() in name.lower():
            stock = ", ".join(f"{n} {kind}" for kind, n in counts.items())
            return f"{name}: {stock}"
    return f"Unknown station '{station}'. Our stations are: {', '.join(STOCK)}."


INTRO = ("You are the help desk of Corniche Cycles, a bike-rental shop in Alexandria. "
         "Answer questions about rental prices and bike availability, briefly and politely.\n"
         "- Never calculate a price yourself: always call rental_quote, which knows the discounts.\n")
FIRST_DRAFT_RULE = "- If the bike type or the number of hours is missing, ask for it instead of guessing.\n"
# The first offline run showed FIRST_DRAFT_RULE firing far too often: the desk
# asked "which bike type?" about a station and about a two-bike comparison.
FIXED_RULE = ("- Ask a follow-up question only when a tool needs something the customer didn't give: "
              "rental_quote needs a bike type and hours, bikes_available needs only a station. "
              "If the customer names several bike types, quote each one.\n")
HANDOFF_AND_SCOPE = (
    "- Repairs, refunds, accidents and complaints go to a person: say that a staff member "
    "will contact them, and do not promise anything.\n"
    "- For anything unrelated to renting bikes, say politely that you can only help with rentals."
)

VERSIONS = {
    "v1": (INTRO + FIRST_DRAFT_RULE + HANDOFF_AND_SCOPE, [rental_quote, bikes_available]),
    "v2": (INTRO + FIXED_RULE + HANDOFF_AND_SCOPE, [rental_quote, bikes_available]),
    "v3": (INTRO + FIXED_RULE + HANDOFF_AND_SCOPE, [rental_quote_refactored, bikes_available]),
}


def build_desk(version: str = "v2"):
    """Compile the help-desk graph for one version: its system prompt and its tools."""
    prompt, tools = VERSIONS[version]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
    system = SystemMessage(prompt)

    def call_llm(state: MessagesState):
        # The system prompt is prepended on every lap, never stored in state.
        return {"messages": [llm.invoke([system] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)  # "tools" or END
    builder.add_edge("tools", "llm")
    return builder.compile()


def run_desk(desk, question: str, config: dict | None = None) -> dict:
    """Run one question and return what an evaluator needs: the answer, the route, the evidence."""
    result = desk.invoke({"messages": [HumanMessage(question)]}, config=config)
    messages = result["messages"]
    tools_called = [call["name"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    return {"answer": messages[-1].content, "tools_called": tools_called, "tool_results": tool_results}
