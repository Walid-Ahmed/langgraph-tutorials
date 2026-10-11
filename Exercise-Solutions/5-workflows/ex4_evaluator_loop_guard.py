# Run from the repository root:
#   python "Exercise-Solutions/5-workflows/ex4_evaluator_loop_guard.py"
#
# What it does:
#   Exercise 5.5, part 1 (and Part 4 Exercise 1 in 5-Workflows/05_evaluator_optimizer.md):
#   adds a round cap to the uncapped support-reply loop in
#   5-Workflows/05_evaluator_optimizer.py. The router gets a third exit,
#   "out_of_rounds", so the loop ends after MAX_ROUNDS reviews even when the
#   reviewer never approves.
#
# What it demonstrates:
#   - the verdict is a stopping signal; the round counter is the guarantee
#   - the counter is incremented where the draft is judged (review_reply), so it
#     counts reviews, not attempts to write
#   - the run uses a very strict 15-word policy, so the cap often has to fire
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies):
#   [round 1] reviewer: revise — The reply is ... words. Cut it to under 15.
#   ...
#   Stopped by: approve or out_of_rounds, never after more than 3 round(s)

from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

MAX_ROUNDS = 3
WORD_LIMIT = 15     # deliberately strict, so the guard often has to fire

REPLY_POLICY = f"""\
1. Name the customer's specific problem.
2. Apologize once, without blaming the customer.
3. Give one concrete next step and when it will happen.
4. Never promise a refund, credit, or compensation.
5. Stay under {WORD_LIMIT} words."""


class State(TypedDict):
    complaint: str
    draft: str
    verdict: str
    critique: str
    rounds: int
    stopped_by: str


class Review(BaseModel):
    verdict: Literal["approve", "revise"] = Field(description="approve only if every rule is followed")
    critique: str = Field(description="If revise: the specific changes to make. If approve: empty.")


reviewer = llm.with_structured_output(Review)


def draft_reply(state: State) -> dict:
    prompt = f"You are a friendly support agent. Reply to this complaint:\n{state['complaint']}"
    if state.get("critique"):
        prompt += (f"\n\nYour previous draft:\n{state['draft']}\n\n"
                   f"Fix these points:\n{state['critique']}")
    return {"draft": llm.invoke(prompt).text}


def review_reply(state: State) -> dict:
    rounds = state["rounds"] + 1      # count reviews here, where the draft is judged
    words = len(state["draft"].split())
    if words > WORD_LIMIT:
        verdict, critique = "revise", f"The reply is {words} words. Cut it to under {WORD_LIMIT}."
    else:
        review = cast(Review, reviewer.invoke(
            f"Policy:\n{REPLY_POLICY}\n\nComplaint: {state['complaint']}\n\nReply:\n{state['draft']}"))
        verdict, critique = review.verdict, review.critique
    print(f"[round {rounds}] reviewer: {verdict}" + (f" — {critique}" if verdict == "revise" else ""))
    return {"verdict": verdict, "critique": critique, "rounds": rounds}


def route_after_review(state: State) -> str:
    if state["verdict"] == "approve":
        return "approve"
    if state["rounds"] >= MAX_ROUNDS:   # the guarantee: checked before looping again
        return "out_of_rounds"
    return "revise"


def mark_stop(label: str):
    """A tiny node that records which exit ended the run."""
    return lambda state: {"stopped_by": label}


builder = StateGraph(State)
builder.add_node("draft_reply", draft_reply)
builder.add_node("review_reply", review_reply)
builder.add_node("approved", mark_stop("approve"))
builder.add_node("gave_up", mark_stop("out_of_rounds"))
builder.add_edge(START, "draft_reply")
builder.add_edge("draft_reply", "review_reply")
builder.add_conditional_edges("review_reply", route_after_review,
                              {"approve": "approved", "revise": "draft_reply",
                               "out_of_rounds": "gave_up"})
builder.add_edge("approved", END)
builder.add_edge("gave_up", END)
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"complaint": "I was charged twice for March and nobody answers my emails.",  # type: ignore[arg-type]
                           "rounds": 0})
    print(f"\nStopped by: {result['stopped_by']} after {result['rounds']} round(s)\n\n{result['draft']}")
