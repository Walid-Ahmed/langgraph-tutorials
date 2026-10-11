# Run from the repository root:
#   python "5-Workflows/05_evaluator_optimizer_max_iterations.py"
#
# What it does:
#   The same support-reply loop as 05_evaluator_optimizer.py (a writer drafts,
#   a reviewer checks the draft against the reply policy, the critique goes
#   back to the writer), with a cap: after MAX_ROUNDS reviews without approval,
#   the draft goes to an escalate node that flags it for a human instead of
#   looping again.
#
# What it demonstrates (Chapter 5 — evaluator–optimizer, the version to ship):
#   - THREE exits from the router: approved, revise, or out of rounds
#   - the reviewer's verdict is a stopping signal; the round cap is a guarantee
#   - running out of rounds is a result to handle (escalate), not a silent END
#     that would send an unapproved reply
#   - the run below uses a strict 40-word policy so you can watch the cap fire
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies; the strict run usually escalates):
#   === Normal policy (120 words) ===   ... status: approved after 1-3 round(s)
#   === Strict policy (40 words)  ===   ... status: escalated to a human after 3 round(s)

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

MAX_ROUNDS = 3


def reply_policy(word_limit: int) -> str:
    return f"""\
1. Name the customer's specific problem, so they know it was read.
2. Apologize once, without blaming the customer.
3. Give one concrete next step and when it will happen.
4. Never promise a refund, credit, or compensation: only the billing team can grant one.
5. Stay under {word_limit} words."""


class State(TypedDict):
    complaint: str
    word_limit: int     # part of the policy, passed in so one graph can run two policies
    draft: str
    verdict: str        # "approve" or "revise"
    critique: str
    rounds: int         # reviews so far; the router's guarantee reads this
    status: str         # how the run ended: "approved" or "escalated"


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
        prompt += (f"\n\nYour previous draft:\n{state['draft']}\n\n"
                   f"A reviewer asked for these changes:\n{state['critique']}\n\n"
                   "Rewrite the reply, keeping what was fine and fixing what was asked.")
    reply = llm.invoke(prompt)
    print(f"  [round {state['rounds'] + 1}] draft: {len(reply.text.split())} words")
    return {"draft": reply.text}


def review_reply(state: State) -> dict:
    """Check the draft: the word limit in code, everything else with the model."""
    rounds = state["rounds"] + 1
    words = len(state["draft"].split())
    if words > state["word_limit"]:
        verdict, critique = "revise", f"The reply is {words} words. Cut it to under {state['word_limit']}."
    else:
        review = cast(Review, reviewer.invoke(
            f"You review customer-support replies against this policy:\n"
            f"{reply_policy(state['word_limit'])}\n\n"
            f"Complaint: {state['complaint']}\n\nReply to review:\n{state['draft']}"))
        verdict, critique = review.verdict, review.critique
    print(f"  [round {rounds}] reviewer: {verdict}" + (f" — {critique}" if verdict == "revise" else ""))
    return {"verdict": verdict, "critique": critique, "rounds": rounds}


def route_after_review(state: State) -> str:
    """Three exits. The verdict is a signal; the round count is the guarantee."""
    if state["verdict"] == "approve":
        return "approve"
    if state["rounds"] >= MAX_ROUNDS:
        return "out_of_rounds"
    return "revise"


def approve(state: State) -> dict:
    return {"status": "approved"}


def escalate(state: State) -> dict:
    """Out of rounds: hand the last draft and the open critique to a person."""
    return {"status": "escalated"}


builder = StateGraph(State)
builder.add_node("draft_reply", draft_reply)
builder.add_node("review_reply", review_reply)
builder.add_node("approve", approve)
builder.add_node("escalate", escalate)
builder.add_edge(START, "draft_reply")
builder.add_edge("draft_reply", "review_reply")
builder.add_conditional_edges("review_reply", route_after_review,
                              {"approve": "approve", "revise": "draft_reply",
                               "out_of_rounds": "escalate"})
builder.add_edge("approve", END)
builder.add_edge("escalate", END)
graph = builder.compile()

COMPLAINT = ("I was charged twice for my March subscription, and nobody has answered my "
             "email for a week. I want my money back today, or I'm cancelling.")


if __name__ == "__main__":
    plot_graph(graph, output_path=str(Path(__file__).parent / "diagrams" /
                                      "05_evaluator_optimizer_max_iterations_graph.png"),
               print_mermaid=False)
    for label, limit in [("Normal policy (120 words)", 120), ("Strict policy (40 words)", 40)]:
        print(f"\n=== {label} ===")
        result = graph.invoke({"complaint": COMPLAINT, "word_limit": limit, "rounds": 0})  # type: ignore[arg-type]
        print(f"  status: {result['status']} after {result['rounds']} round(s)")
        if result["status"] == "escalated":
            print(f"  open critique for the human: {result['critique']}")
        print(f"\n{result['draft']}")
