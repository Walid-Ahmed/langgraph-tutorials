# Run from the repository root:
#   python "5-Workflows/05_evaluator_optimizer.py"
#
# What it does:
#   Drafts a reply to an upset customer, then has a reviewer check the draft
#   against the company's reply policy. If the reviewer asks for changes, the
#   draft goes back to the writer with the critique, and the loop repeats until
#   the reviewer approves.
#
# What it demonstrates (Chapter 5 — evaluator–optimizer):
#   - the only Chapter 5 pattern with a cycle: draft -> review -> (revise) -> review ...
#   - the evaluator's verdict arrives as structured output, so the router can branch on it
#   - the critique travels in state, and the writer revises its previous draft
#     instead of starting over
#   - the reviewer can be code AND a model: the word limit is counted in code,
#     the judgment calls go to the model
#   - this version has NO round cap: it stops only when the reviewer approves
#     (or LangGraph's recursion limit fires). 05_evaluator_optimizer_max_iterations.py
#     adds the cap you'd ship.
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies):
#   [round 1] draft: ~90 words
#   [round 1] reviewer: revise — e.g. "Don't promise a refund; only billing can issue one."
#   [round 2] draft: ...
#   [round 2] reviewer: approve
#   followed by the approved reply. Most runs approve on round 2 or 3.
#
# Then run:
#   python "5-Workflows/05_evaluator_optimizer_max_iterations.py"

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

llm = ChatOpenAI(model="gpt-4o", temperature=0)

WORD_LIMIT = 120

# The reviewer owns the policy. The writer never sees it, which is realistic:
# a support agent writes, a team lead checks against the house rules.
REPLY_POLICY = f"""\
1. Name the customer's specific problem, so they know it was read.
2. Apologize once, without blaming the customer.
3. Give one concrete next step and when it will happen.
4. Never promise a refund, credit, or compensation: only the billing team can grant one.
5. Stay under {WORD_LIMIT} words."""


class State(TypedDict):
    complaint: str      # set once by the caller
    draft: str          # written by draft_reply, read by review_reply
    verdict: str        # "approve" or "revise", written by review_reply, read by the router
    critique: str       # what to fix, written by review_reply, read by draft_reply
    rounds: int         # how many drafts have been reviewed


class Review(BaseModel):
    verdict: Literal["approve", "revise"] = Field(
        description="approve only if the reply follows every rule in the policy")
    critique: str = Field(
        description="If revise: the specific changes to make, as instructions. If approve: empty.")


reviewer = llm.with_structured_output(Review)


def draft_reply(state: State) -> dict:
    """Write the first draft, or revise the last one using the reviewer's critique."""
    prompt = ("You are a friendly customer-support agent. Write a reply to this complaint.\n\n"
              f"Complaint: {state['complaint']}")
    if state.get("critique"):
        # Revise, don't restart: the writer sees its own draft next to the critique.
        prompt += (f"\n\nYour previous draft:\n{state['draft']}\n\n"
                   f"A reviewer asked for these changes:\n{state['critique']}\n\n"
                   "Rewrite the reply, keeping what was fine and fixing what was asked.")
    reply = llm.invoke(prompt)
    print(f"[round {state.get('rounds', 0) + 1}] draft: {len(reply.text.split())} words")
    return {"draft": reply.text}


def review_reply(state: State) -> dict:
    """Check the draft: the word limit in code, everything else with the model."""
    rounds = state.get("rounds", 0) + 1
    words = len(state["draft"].split())
    if words > WORD_LIMIT:
        # A rule code can check exactly: don't pay a model to count words.
        verdict, critique = "revise", f"The reply is {words} words. Cut it to under {WORD_LIMIT}."
    else:
        review = cast(Review, reviewer.invoke(
            f"You review customer-support replies against this policy:\n{REPLY_POLICY}\n\n"
            f"Complaint: {state['complaint']}\n\nReply to review:\n{state['draft']}"))
        verdict, critique = review.verdict, review.critique
    print(f"[round {rounds}] reviewer: {verdict}" + (f" — {critique}" if verdict == "revise" else ""))
    return {"verdict": verdict, "critique": critique, "rounds": rounds}


def route_after_review(state: State) -> str:
    """Two exits: approved, or back to the writer with the critique in state."""
    return "approve" if state["verdict"] == "approve" else "revise"


builder = StateGraph(State)
builder.add_node("draft_reply", draft_reply)
builder.add_node("review_reply", review_reply)
builder.add_edge(START, "draft_reply")
builder.add_edge("draft_reply", "review_reply")
builder.add_conditional_edges("review_reply", route_after_review,
                              {"approve": END, "revise": "draft_reply"})
graph = builder.compile()

COMPLAINT = ("I was charged twice for my March subscription, and nobody has answered my "
             "email for a week. I want my money back today, or I'm cancelling.")


if __name__ == "__main__":
    plot_graph(graph, output_path=str(Path(__file__).parent / "diagrams" / "05_evaluator_optimizer_graph.png"),
               print_mermaid=False)
    result = graph.invoke({"complaint": COMPLAINT, "rounds": 0})  # type: ignore[arg-type]
    print(f"\nApproved after {result['rounds']} round(s):\n\n{result['draft']}")
