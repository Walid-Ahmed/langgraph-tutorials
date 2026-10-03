# Step 11 in 6-Agents: two agents side by side, each a subgraph of one parent graph.
#
# This script sends ONE question to TWO create_agent specialists at the same
# time. A weather agent and a budget agent each work with their own tools, each
# writes its report into its own field of the parent state, and a final
# combine node (a plain LLM call, no tools) merges the two reports.
#
#     START -> weather --\
#                         >-> combine -> END
#     START -> budget  --/
#
# Why this matters:
# - A compiled graph run inside another graph is a SUBGRAPH. Each agent keeps
#   its own private state (its `messages` list); the parent never sees it.
# - The two lines inside each agent node are the whole boundary: one maps the
#   parent's `question` into the agent's `messages`, one maps the agent's last
#   message back into a single parent field.
# - Each parallel agent owns its own field, so no reducer is needed (Chapter 5).
# - It does NOT let a model choose which agent runs: the edges decide, so both
#   always run. An agent that calls other agents as tools is a "subagent"
#   design; see Exercise 8.4 in the book.
#
# Before you run this:
# 1. Put OPENAI_API_KEY=... in the repo-root .env
# 2. pip install -r requirements.txt
# 3. The weather tool calls the free Open-Meteo API (no key). To skip that
#    call and use a fixed sample forecast, set TRIP_SAMPLE_WEATHER=1.
#
# Run it from the repository root with:
#    python "6-Agents/11_parallel_subgraphs.py"
# or, with the sample forecast:
#    TRIP_SAMPLE_WEATHER=1 python "6-Agents/11_parallel_subgraphs.py"
#
# Then run:
#    python "6-Agents/12_subgraph_shared_keys.py"   (the other way to attach a subgraph)

import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from trip_tools import BUDGET_TOOLS, WEATHER_TOOLS  # noqa: E402
from util import plot_graph  # noqa: E402

QUESTION = ("Should I visit Montreal this weekend? "
            "I would take the train from Toronto and stay one night.")


# ---------------------------------------------------------
# 1. The two subgraphs
#
# Each create_agent call returns a compiled graph whose only state key is
# `messages`. Different tools and a narrow prompt keep each one on its part
# of the question.
# ---------------------------------------------------------
weather_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=WEATHER_TOOLS,
    system_prompt=("You are a weather specialist. Report only the weather for the "
                   "trip in the question, in at most three sentences. "
                   "Always use your tool; never guess a forecast."),
)

budget_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=BUDGET_TOOLS,
    system_prompt=("You are a travel budget specialist. Report only the cost of the "
                   "trip in the question: look up each price, add them with add_costs, "
                   "and give the total in at most three sentences."),
)

# The combiner has no tools: its whole job is to read two reports.
combiner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------
# 2. The parent state: one baton, four boxes, one writer per box
#
# There is no `messages` key here on purpose. The agents' tool calls stay
# inside the agents.
# ---------------------------------------------------------
class TripState(TypedDict):
    question: str          # set by the caller; both agents read it
    weather_report: str    # written only by weather_node
    budget_report: str     # written only by budget_node
    final_answer: str      # written only by combine_node


def tools_executed(messages: list) -> list[str]:
    names = [m.name for m in messages if isinstance(m, ToolMessage) and m.name]
    return list(dict.fromkeys(names))


# ---------------------------------------------------------
# 3. The nodes: each one runs a subgraph and translates between the two states
# ---------------------------------------------------------
def weather_node(state: TripState) -> dict:
    result = weather_agent.invoke(
        {"messages": [HumanMessage(state["question"])]}        # parent -> subgraph
    )
    print("Weather agent tools:", tools_executed(result["messages"]) or "(none)")
    return {"weather_report": result["messages"][-1].content}  # subgraph -> parent


def budget_node(state: TripState) -> dict:
    result = budget_agent.invoke(
        {"messages": [HumanMessage(state["question"])]}
    )
    print("Budget agent tools:", tools_executed(result["messages"]) or "(none)")
    return {"budget_report": result["messages"][-1].content}


def combine_node(state: TripState) -> dict:
    prompt = (f"Question:\n{state['question']}\n\n"
              f"Weather report:\n{state['weather_report']}\n\n"
              f"Budget report:\n{state['budget_report']}\n\n"
              "Give one recommendation in at most four sentences, "
              "using both reports and nothing else.")
    return {"final_answer": combiner_llm.invoke(prompt).content}


# ---------------------------------------------------------
# 4. The wiring: fork, then merge
# ---------------------------------------------------------
builder = StateGraph(TripState)
builder.add_node("weather", weather_node)
builder.add_node("budget", budget_node)
builder.add_node("combine", combine_node)
for agent_name in ("weather", "budget"):
    builder.add_edge(START, agent_name)        # fan-out: both get the same question
    builder.add_edge(agent_name, "combine")    # fan-in: combine waits for both
builder.add_edge("combine", END)
graph = builder.compile()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repo-root .env.")

    png = Path(__file__).resolve().parent / "diagrams" / "11_parallel_subgraphs.png"
    png.parent.mkdir(exist_ok=True)
    plot_graph(graph, str(png), print_mermaid=False)

    print("Question:", QUESTION, "\n")
    # stream_mode="updates" shows which node wrote which field, step by step.
    final_answer = ""
    for update in graph.stream({"question": QUESTION}, stream_mode="updates"):
        for node_name, written in update.items():
            for field, value in written.items():
                print(f"\n[{node_name}] wrote {field}:\n{value}")
                if field == "final_answer":
                    final_answer = value

    # Expected shape: two tool lines, then [weather] and [budget] (either order,
    # same step), then [combine] last. The wording varies from run to run.
    if not final_answer:
        raise SystemExit("combine did not produce a final answer.")


if __name__ == "__main__":
    main()
