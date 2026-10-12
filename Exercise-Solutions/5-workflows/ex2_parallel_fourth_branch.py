# Run from the repository root:
#   python "Exercise-Solutions/5-workflows/ex2_parallel_fourth_branch.py"
#
# What it does:
#   Exercise 5.3: adds a fourth parallel writer, a newsletter blurb, to the
#   social-media example in 5-Workflows/03_parallelization.py. The new branch
#   needs a new state field, a new node, one fan-out edge from START and one
#   fan-in edge into the aggregator.
#
# What it demonstrates:
#   - adding a branch to a static fan-out touches the state, the nodes and two edges
#   - the new writer owns a distinct field, so the four writers run in the same
#     step with nothing to merge and no reducer
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies):
#   INSTAGRAM ... TWITTER/X ... LINKEDIN ... NEWSLETTER (four sections)

from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class OverallState(TypedDict):
    topic: str
    instagram_post: str
    twitter_post: str
    linkedin_post: str
    newsletter_blurb: str      # new: a distinct field for the fourth writer
    final_output: str


# The three original writers, condensed from 03_parallelization.py.
PROMPTS = {
    "instagram_post": "Write an Instagram post with 3 hashtags about: {topic}",
    "twitter_post": "Write one post under 280 characters about: {topic}",
    "linkedin_post": "Write a short professional LinkedIn post about: {topic}",
}


def make_writer(field: str):
    def writer(state: OverallState) -> dict:
        return {field: llm.invoke(PROMPTS[field].format(topic=state["topic"])).text}
    return writer


# The exercise: one new node that writes one new field.
def generate_newsletter_blurb(state: OverallState) -> dict:
    prompt = f"Write a 2-sentence blurb for an email newsletter about: {state['topic']}"
    return {"newsletter_blurb": llm.invoke(prompt).text}


def aggregate_posts(state: OverallState) -> dict:
    sections = [("INSTAGRAM", "instagram_post"), ("TWITTER/X", "twitter_post"),
                ("LINKEDIN", "linkedin_post"), ("NEWSLETTER", "newsletter_blurb")]
    return {"final_output": "\n\n".join(f"{title}\n{state[field]}" for title, field in sections)}


builder = StateGraph(OverallState)
builder.add_node("aggregate_posts", aggregate_posts)
for field in PROMPTS:
    builder.add_node(f"generate_{field}", make_writer(field))
    builder.add_edge(START, f"generate_{field}")
    builder.add_edge(f"generate_{field}", "aggregate_posts")

builder.add_node("generate_newsletter_blurb", generate_newsletter_blurb)
builder.add_edge(START, "generate_newsletter_blurb")              # fan-out
builder.add_edge("generate_newsletter_blurb", "aggregate_posts")  # fan-in
builder.add_edge("aggregate_posts", END)
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"topic": "The impact of AI on workplace productivity"})  # type: ignore[arg-type]
    print(result["final_output"])
