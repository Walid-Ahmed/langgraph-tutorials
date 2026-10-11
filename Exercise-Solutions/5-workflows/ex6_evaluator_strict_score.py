# Run from the repository root:
#   python "Exercise-Solutions/5-workflows/ex6_evaluator_strict_score.py"
#
# What it does:
#   Exercise 5.5, part 2 (and Part 4 Exercise 3 in 5-Workflows/05_evaluator_optimizer.md):
#   the bounded support-reply loop, but the reviewer returns a numeric score
#   from 1 to 10 instead of approve/revise. Code turns the score into a
#   decision: approve only at PASS_SCORE or above, and stop after MAX_ROUNDS.
#
# What it demonstrates:
#   - a numeric score gives you a threshold you can tune and a trend you can log
#   - the threshold lives in code (PASS_SCORE), not in the prompt, so changing
#     the bar never means rewording what you ask the model
#   - the round cap still guarantees the loop ends
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies):
#   [round 1] score 6/10 — ...
#   [round 2] score 8/10
#   Scores by round: [6, 8] -> approved

from typing import TypedDict, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

MAX_ROUNDS = 3
PASS_SCORE = 8

REPLY_POLICY = """\
1. Name the customer's specific problem.
2. Apologize once, without blaming the customer.
3. Give one concrete next step and when it will happen.
4. Never promise a refund, credit, or compensation.
5. Stay under 120 words."""


class State(TypedDict):
    complaint: str
    draft: str
    score: int
    critique: str
    scores: list[int]   # one per round, so you can see the trend
    rounds: int


class ScoredReview(BaseModel):
    score: int = Field(ge=1, le=10, description="How well the reply follows the policy, 1-10")
    critique: str = Field(description="The changes that would raise the score")


reviewer = llm.with_structured_output(ScoredReview)


def draft_reply(state: State) -> dict:
    prompt = f"You are a friendly support agent. Reply to this complaint:\n{state['complaint']}"
    if state.get("critique"):
        prompt += f"\n\nYour previous draft:\n{state['draft']}\n\nFix these points:\n{state['critique']}"
    return {"draft": llm.invoke(prompt).text}


def review_reply(state: State) -> dict:
    review = cast(ScoredReview, reviewer.invoke(
        f"Score this reply against the policy.\n\nPolicy:\n{REPLY_POLICY}\n\n"
        f"Complaint: {state['complaint']}\n\nReply:\n{state['draft']}"))
    rounds = state["rounds"] + 1
    print(f"[round {rounds}] score {review.score}/10"
          + (f" — {review.critique}" if review.score < PASS_SCORE else ""))
    return {"score": review.score, "critique": review.critique,
            "scores": state["scores"] + [review.score], "rounds": rounds}


def route_after_review(state: State) -> str:
    if state["score"] >= PASS_SCORE:      # the bar, in code
        return "approve"
    if state["rounds"] >= MAX_ROUNDS:
        return "out_of_rounds"
    return "revise"


builder = StateGraph(State)
builder.add_node("draft_reply", draft_reply)
builder.add_node("review_reply", review_reply)
builder.add_edge(START, "draft_reply")
builder.add_edge("draft_reply", "review_reply")
builder.add_conditional_edges("review_reply", route_after_review,
                              {"approve": END, "out_of_rounds": END, "revise": "draft_reply"})
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"complaint": "I was charged twice for March, and nobody has answered "  # type: ignore[arg-type]
                                        "my email for a week. I want my money back today.",
                           "scores": [], "rounds": 0})
    outcome = "approved" if result["score"] >= PASS_SCORE else "out of rounds"
    print(f"\nScores by round: {result['scores']} -> {outcome}\n\n{result['draft']}")
