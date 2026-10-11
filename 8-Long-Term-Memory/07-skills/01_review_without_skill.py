# Run from the repository root:
#   python "8-Long-Term-Memory/07-skills/01_review_without_skill.py"
#
# What it does:
#   Reviews a short paper abstract with a one-node graph. The whole review
#   method (seven sections, from summary to recommendation) is written
#   directly into the system prompt.
#
# What it demonstrates (Chapter 11 — skills):
#   - the baseline: procedural knowledge baked into the system prompt
#   - it works, but the method is tied to this one agent; a second kind of
#     task means a longer prompt that the model reads on every call
#   - 02_review_with_skill.py produces the same review with the method in a file
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies):
#   System prompt: about 140 words
#   A review with seven headed sections, ending in a recommendation such as
#   "Weak Accept" or "Weak Reject" with a two-sentence justification.
#
# Then run:
#   python "8-Long-Term-Memory/07-skills/02_review_with_skill.py"

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

model = init_chat_model("openai:gpt-4o-mini", temperature=0)

PAPER = """
We propose AQR, a retrieval-augmented generation method that rewrites each user
query before retrieval. A small model generates three rewrites and the union of
their results is passed to the generator. On two open-domain QA benchmarks, AQR
improves top-5 retrieval accuracy over BM25 and a dense retriever.
"""


class ReviewState(TypedDict):
    paper: str
    review: str


# The method lives here, inside the prompt: every call carries all of it.
SYSTEM_PROMPT = """You are an expert academic paper reviewer.
Write the review in these seven sections, in this order, each with its heading.

1. Summary: the problem, the method, and the main contributions, in three sentences or fewer.
2. Novelty: how original the contribution is and how it differs from prior work.
3. Technical soundness: assumptions, method, experiments, and whether evidence supports the claims.
4. Strengths: the strongest aspects of the work.
5. Weaknesses: limitations, missing experiments, and unsupported claims.
6. Questions for the authors: exactly three constructive questions.
7. Recommendation: Accept, Weak Accept, Weak Reject, or Reject, justified in two sentences.

Base every judgment on the text supplied. Never invent results, numbers, or
experiments the paper doesn't report; if something is missing, say it's missing."""


def review_paper(state: ReviewState) -> dict:
    """Send the method (system prompt) and the paper (user message) to the model."""
    reply = model.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(state["paper"])])
    return {"review": reply.content}


builder = StateGraph(ReviewState)
builder.add_node("review_paper", review_paper)
builder.add_edge(START, "review_paper")
builder.add_edge("review_paper", END)
graph = builder.compile()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repository-root .env file.")

    print(f"System prompt: {len(SYSTEM_PROMPT.split())} words\n")
    result = graph.invoke({"paper": PAPER, "review": ""})
    print(result["review"])


if __name__ == "__main__":
    main()
