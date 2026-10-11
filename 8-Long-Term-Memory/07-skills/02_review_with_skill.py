# Run from the repository root:
#   python "8-Long-Term-Memory/07-skills/02_review_with_skill.py"
#
# What it does:
#   Produces the same paper review as 01_review_without_skill.py, but the review
#   method now lives in a file, skills/paper_review/SKILL.md. A load_skill node
#   reads the file into the state; the review_paper node puts it in front of the
#   model next to a short, general system prompt.
#
# What it demonstrates (Chapter 11 — skills):
#   - a skill is procedural memory kept in a file: written by people, versioned
#     with the code, the same for every user
#   - loading a skill is an ordinary node; LangGraph has no special primitive for it
#   - putting text in the state does NOT put it in the model's context:
#     review_paper has to add skill_instructions to the messages itself
#   - this graph always loads the same skill; 03_agent_loads_skill.py lets the
#     agent choose one from a catalog
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies):
#   [skill] loaded skills/paper_review/SKILL.md (about 190 words)
#   A review with the same seven headed sections as 01, ending in a
#   recommendation with a two-sentence justification.
#
# Then run:
#   python "8-Long-Term-Memory/07-skills/03_agent_loads_skill.py"

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")

model = init_chat_model("openai:gpt-4o-mini", temperature=0)

PAPER = """
We propose AQR, a retrieval-augmented generation method that rewrites each user
query before retrieval. A small model generates three rewrites and the union of
their results is passed to the generator. On two open-domain QA benchmarks, AQR
improves top-5 retrieval accuracy over BM25 and a dense retriever.
"""


class ReviewState(TypedDict):
    paper: str
    skill_instructions: str
    review: str


# Short and general: the method is no longer part of the agent's own prompt.
SYSTEM_PROMPT = (
    "You are a professional research assistant. "
    "Follow the specialized instructions below when performing the task."
)


def load_skill(state: ReviewState) -> dict:
    """Read the skill file and return it as a partial state update."""
    skill_path = HERE / "skills" / "paper_review" / "SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    print(f"[skill] loaded {skill_path.relative_to(HERE)} ({len(text.split())} words)\n")
    return {"skill_instructions": text}


def review_paper(state: ReviewState) -> dict:
    """Build the model's context from the general prompt plus the loaded skill."""
    # The skill reaches the model only because this line adds it to the messages.
    system = SYSTEM_PROMPT + "\n\nSpecialized instructions:\n" + state["skill_instructions"]
    reply = model.invoke([SystemMessage(system), HumanMessage(state["paper"])])
    return {"review": reply.content}


builder = StateGraph(ReviewState)
builder.add_node("load_skill", load_skill)
builder.add_node("review_paper", review_paper)
builder.add_edge(START, "load_skill")
builder.add_edge("load_skill", "review_paper")
builder.add_edge("review_paper", END)
graph = builder.compile()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repository-root .env file.")

    result = graph.invoke({"paper": PAPER, "skill_instructions": "", "review": ""})
    print(result["review"])


if __name__ == "__main__":
    main()
