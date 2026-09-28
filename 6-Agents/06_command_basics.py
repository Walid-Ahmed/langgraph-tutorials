# Run from the repository root:
#   python "6-Agents/06_command_basics.py"
#
# What it does:
#   A tiny help-desk graph with NO model and NO API key. A `triage` node reads
#   the request, updates state, and chooses the next node in one step by
#   returning Command(update=..., goto=...). Then a `billing` desk hands off to
#   a `refunds` desk the same way. There is not a single add_conditional_edges
#   call in the file.
#
# What it demonstrates (Chapter 8 — Agents, §8.4 "Command"):
#   - Command = a state update AND the next hop, returned together by a node
#   - the Command[Literal[...]] return annotation, which tells LangGraph where
#     a node may go (so compile() validates it and the drawn graph is complete)
#   - a handoff: one "desk" passing the request to another, the same pattern
#     multi-agent systems use to pass control between agents
#
# Prerequisites: none beyond `pip install langgraph` (no API key needed).
#
# Expected output (deterministic):
#   'My invoice is wrong'        -> triage -> billing -> END
#   'Please refund my order'     -> triage -> billing -> refunds -> END
#   'The app crashes on login'   -> triage -> tech -> END
#   plus the list of edges LangGraph derived from the Command annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict


class State(TypedDict):
    request: str
    desk: str          # which desk is handling the request now
    path: list[str]    # every desk that touched it, in order
    reply: str
    # No reducer: each node returns a full new `path` list (overwrite).


def triage(state: State) -> Command[Literal["billing", "tech"]]:
    """Decide where the request goes: update state AND route, in one return."""
    # Literal[...] is for compile() and the drawn graph, not a runtime switch.
    # The if/else below is what actually picks billing vs tech.
    text = state["request"].lower()
    desk = "billing" if any(w in text for w in ("invoice", "refund", "charge")) else "tech"
    return Command(
        update={"desk": desk, "path": ["triage"]},   # the state update...
        goto=desk,                                   # ...and the next node
    )


def billing(state: State) -> Command[Literal["refunds", "__end__"]]:
    """Answer billing questions, or hand refunds off to the refunds desk."""
    # In Command[Literal], END is written as the string "__end__".
    if "refund" in state["request"].lower():
        # A handoff: billing passes control (and a note) to another desk.
        return Command(
            update={"desk": "refunds", "path": state["path"] + ["billing"]},
            goto="refunds",
        )
    return Command(
        update={"path": state["path"] + ["billing"],
                "reply": "Billing: we have corrected your invoice."},
        goto=END,
    )


def refunds(state: State) -> dict:
    """A plain node: ordinary edges and ordinary returns still work."""
    return {"path": state["path"] + ["refunds"], "reply": "Refunds: your refund is on its way."}


def tech(state: State) -> dict:
    """Same as refunds: return a state patch; the graph edge goes to END."""
    return {"path": state["path"] + ["tech"], "reply": "Tech: please update the app and retry."}


builder = StateGraph(State)
builder.add_node("triage", triage)
builder.add_node("billing", billing)
builder.add_node("refunds", refunds)
builder.add_node("tech", tech)
builder.add_edge(START, "triage")
builder.add_edge("refunds", END)   # plain nodes still need plain edges
builder.add_edge("tech", END)
# No add_conditional_edges: triage and billing route themselves with Command.
graph = builder.compile()


def main() -> None:
    for request in ["My invoice is wrong", "Please refund my order", "The app crashes on login"]:
        result = graph.invoke({"request": request, "desk": "", "path": [], "reply": ""})
        print(f"{request!r:28} -> {' -> '.join(result['path'])} -> END")
        print(f"{'':28}    {result['reply']}")

    print("\nEdges LangGraph derived from the Command[Literal[...]] annotations:")
    for edge in graph.get_graph().edges:
        print(f"  {edge.source} -> {edge.target}{'   (via Command)' if edge.conditional else ''}")


if __name__ == "__main__":
    main()
