# Run from the repository root:
#   python "Exercise-Solutions/5-workflows/ex5_evaluator_headline_writer.py"
#
# What it does:
#   Part 4 Exercise 2 in 5-Workflows/05_evaluator_optimizer.md: the same
#   evaluator–optimizer loop applied to a different job. A writer drafts a
#   headline for a short article; a reviewer checks it and returns "clear" or
#   "unclear" with a critique; the loop stops on "clear" or after MAX_ROUNDS.
#
# What it demonstrates:
#   - the loop's shape doesn't change when the domain does: only the two
#     prompts, the verdict labels and the code checks change
#   - two code checks (length, banned clickbait phrases) run before the model
#     is asked the one question that needs judgment: is it clear and accurate?
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies):
#   [round 1] headline: "..."  -> unclear / clear
#   ...
#   Final headline (clear after N round(s)): ...

from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)   # some variety in headline wording
judge = ChatOpenAI(model="gpt-4o", temperature=0)

MAX_ROUNDS = 4
MAX_WORDS = 12
BANNED = ("you won't believe", "shocking", "game-changer", "this one trick")

ARTICLE = ("The city council voted 7-2 on Tuesday to extend the downtown bike lane by "
           "3 km, funded by a state grant. Construction starts in May and closes one car "
           "lane on Main Street for six weeks.")


class State(TypedDict):
    article: str
    headline: str
    grade: str
    critique: str
    rounds: int


class HeadlineReview(BaseModel):
    grade: Literal["clear", "unclear"] = Field(
        description="clear only if a reader knows what happened from the headline alone, "
                    "and nothing in it is wrong or exaggerated")
    critique: str = Field(description="If unclear: what to change. If clear: empty.")


headline_judge = judge.with_structured_output(HeadlineReview)


def draft_headline(state: State) -> dict:
    prompt = f"Write one news headline for this article. Reply with the headline only.\n\n{state['article']}"
    if state.get("critique"):
        prompt += f"\n\nYour previous headline: {state['headline']}\nFix this: {state['critique']}"
    return {"headline": llm.invoke(prompt).text.strip().strip('"')}


def review_headline(state: State) -> dict:
    rounds = state["rounds"] + 1
    headline = state["headline"]
    words = len(headline.split())
    hit = next((phrase for phrase in BANNED if phrase in headline.lower()), None)
    if words > MAX_WORDS:
        grade, critique = "unclear", f"{words} words is too long; use at most {MAX_WORDS}."
    elif hit:
        grade, critique = "unclear", f"Drop the clickbait phrase '{hit}'."
    else:
        review = cast(HeadlineReview, headline_judge.invoke(
            f"Article: {state['article']}\n\nHeadline: {headline}"))
        grade, critique = review.grade, review.critique
    print(f'[round {rounds}] headline: "{headline}"  -> {grade}' + (f" ({critique})" if critique else ""))
    return {"grade": grade, "critique": critique, "rounds": rounds}


def route(state: State) -> str:
    if state["grade"] == "clear" or state["rounds"] >= MAX_ROUNDS:
        return "done"
    return "revise"


builder = StateGraph(State)
builder.add_node("draft_headline", draft_headline)
builder.add_node("review_headline", review_headline)
builder.add_edge(START, "draft_headline")
builder.add_edge("draft_headline", "review_headline")
builder.add_conditional_edges("review_headline", route, {"done": END, "revise": "draft_headline"})
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"article": ARTICLE, "rounds": 0})  # type: ignore[arg-type]
    print(f"\nFinal headline ({result['grade']} after {result['rounds']} round(s)): {result['headline']}")
