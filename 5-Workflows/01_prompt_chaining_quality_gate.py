# Run from the repository root:
#   python "5-Workflows/01_prompt_chaining_quality_gate.py"
#
# What it does:
#   Writes a product description for an online store in a chain of model calls,
#   with a quality gate after the first draft. The gate is plain code: the
#   draft passes if it's at most 50 words and mentions the price. A passing
#   draft ends the run early; a failing one continues through two more calls,
#   tighten_description and add_call_to_action.
#
# What it demonstrates (Chapter 5 — prompt chaining with a quality gate):
#   - a chain that can stop early: a conditional edge after the first node
#   - the gate is a router that only reads state, here with a deterministic check
#   - two patterns composing: Chapter 4's branch inside Chapter 5's chain
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies; which path runs depends on the first draft):
#   Draft: ~70 words, price mentioned: True -> gate: fail -> tightened -> with call to action
#   or
#   Draft: ~40 words, price mentioned: True -> gate: pass (ends early)

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

MAX_WORDS = 50


class State(TypedDict):
    product: str
    price: str
    draft: str          # written once by draft_description
    tightened: str      # written once by tighten_description (only on the fail path)
    final: str          # written once by add_call_to_action (only on the fail path)


def draft_description(state: State) -> dict:
    reply = llm.invoke(f"Write a product description for an online store.\n"
                       f"Product: {state['product']}\nPrice: {state['price']}")
    return {"draft": reply.text}


def meets_brief(state: State) -> str:
    """The gate: a router, so it only reads state. Two checks code can do exactly."""
    words = len(state["draft"].split())
    has_price = state["price"] in state["draft"]
    print(f"Draft: {words} words, price mentioned: {has_price}")
    return "pass" if words <= MAX_WORDS and has_price else "fail"


def tighten_description(state: State) -> dict:
    reply = llm.invoke(f"Cut this to at most {MAX_WORDS - 10} words. Keep the price "
                       f"({state['price']}) and the one most useful detail.\n\n{state['draft']}")
    return {"tightened": reply.text}


def add_call_to_action(state: State) -> dict:
    reply = llm.invoke("End this product description with one short call to action. "
                       f"Return the whole description.\n\n{state['tightened']}")
    return {"final": reply.text}


builder = StateGraph(State)
builder.add_node("draft_description", draft_description)
builder.add_node("tighten_description", tighten_description)
builder.add_node("add_call_to_action", add_call_to_action)
builder.add_edge(START, "draft_description")
builder.add_conditional_edges("draft_description", meets_brief,
                              {"pass": END, "fail": "tighten_description"})
builder.add_edge("tighten_description", "add_call_to_action")
builder.add_edge("add_call_to_action", END)
graph = builder.compile()


def main() -> None:
    plot_graph(graph, str(Path(__file__).resolve().parent / "diagrams" / "01_prompt_chaining_quality_gate_graph.png"),
               print_mermaid=False)
    result = graph.invoke({"product": "Insulated steel water bottle, 750 ml, keeps drinks cold for 24 hours",  # type: ignore[arg-type]
                           "price": "$29"})
    if result.get("final"):
        print("Gate: fail -> tightened -> with call to action\n")
        print(result["final"])
    else:
        print("Gate: pass (the chain ended after the first call)\n")
        print(result["draft"])


if __name__ == "__main__":
    main()
