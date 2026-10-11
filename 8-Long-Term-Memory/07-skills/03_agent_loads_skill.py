# Run from the repository root:
#   python "8-Long-Term-Memory/07-skills/03_agent_loads_skill.py"
#
# What it does:
#   Gives a ReAct agent a small skill library (skills/*/SKILL.md). The system
#   prompt carries only a catalog: each skill's name and one-line description.
#   When a request matches a skill, the agent calls the load_skill tool, reads
#   the full instructions, and follows them. Three requests are run: a paper
#   review, a proposal evaluation, and a plain question that needs no skill.
#
# What it demonstrates (Chapter 11 — skills):
#   - progressive disclosure: the catalog is always in the prompt; a skill's
#     full text enters the context only when the agent asks for it
#   - choosing WHICH method fits a request is a judgment the model is good at;
#     knowing that skills exist is not left to it (the catalog is read in code)
#   - the model picks a skill by name, so load_skill accepts only catalog names
#     and never builds a file path from the model's text
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies; which tool calls happen is the model's choice):
#   Catalog in the system prompt: ~45 words; all three skills in full: ~490 words
#   Request 1 -> [skill] loaded paper-review     -> seven headed review sections
#   Request 2 -> [skill] loaded proposal-evaluation -> five 1-5 scores and an overall score
#   Request 3 -> (no skill loaded)               -> a one-line answer

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")

SKILLS_DIR = HERE / "skills"


def read_skill_file(path: Path) -> tuple[dict[str, str], str]:
    """Split a SKILL.md into its front matter (name, description) and its body."""
    _, front_matter, body = path.read_text(encoding="utf-8").split("---", 2)
    fields = dict(line.split(":", 1) for line in front_matter.strip().splitlines())
    return {key.strip(): value.strip() for key, value in fields.items()}, body.strip()


# Read once at startup, in code: names and descriptions only, never the bodies.
CATALOG: dict[str, dict] = {}
for skill_file in sorted(SKILLS_DIR.glob("*/SKILL.md")):
    fields, _ = read_skill_file(skill_file)          # (front matter, body)
    CATALOG[fields["name"]] = {"description": fields["description"], "path": skill_file}


@tool
def load_skill(name: str, state: Annotated[dict, InjectedState]) -> str:
    """Load the full instructions for one skill from the catalog in your system prompt.
    Call this before starting a task that matches a skill, then follow what it returns."""
    # The name comes from the model. Only catalog entries can be opened, so a
    # made-up name (or a path like "../../.env") can't reach any other file.
    if name not in CATALOG:
        return f"Unknown skill '{name}'. Available skills: {', '.join(CATALOG)}."
    loaded_marker = f"Skill '{name}' is now loaded."
    # The guarantee: gpt-4o-mini sometimes asks for the same skill twice. If an
    # earlier tool result in this conversation already holds it, don't send it again.
    if any(m.type == "tool" and m.content.startswith(loaded_marker) for m in state["messages"]):
        print(f"  [skill] {name} already loaded; sent a reminder instead")
        return f"Skill '{name}' is already loaded above. Apply it to the request now."
    print(f"  [skill] loaded {name}")
    _, body = read_skill_file(CATALOG[name]["path"])
    # The signal: tell the model the skill is in its context, which cuts most repeats.
    return f"{loaded_marker} Don't load it again; apply it to the request.\n\n{body}"


catalog_lines = "\n".join(f"- {name}: {entry['description']}" for name, entry in CATALOG.items())
SYSTEM_PROMPT = (
    "You are a research assistant.\n"
    "You have these skills. Each is a written method for one kind of task:\n"
    f"{catalog_lines}\n\n"
    "When a request matches a skill, call load_skill with its name before answering, "
    "then follow the loaded instructions exactly. If no skill matches, answer directly."
)

agent = create_agent(
    model=init_chat_model("openai:gpt-4o-mini", temperature=0),
    tools=[load_skill],
    system_prompt=SYSTEM_PROMPT,
)

REQUESTS = [
    "Please review this paper abstract: We propose AQR, a retrieval-augmented "
    "generation method that rewrites each user query before retrieval. A small model "
    "generates three rewrites and the union of their results is passed to the "
    "generator. On two open-domain QA benchmarks, AQR improves top-5 retrieval "
    "accuracy over BM25 and a dense retriever.",
    "Can you evaluate this grant proposal? Aim: build an open benchmark of 5,000 "
    "annotated legal questions in Arabic and English. Team: two NLP researchers and "
    "one lawyer. Timeline: 12 months. Budget: annotation contractors and cloud credits.",
    "What does BM25 stand for? One sentence, please.",
]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repository-root .env file.")

    all_bodies = sum(len(read_skill_file(e["path"])[1].split()) for e in CATALOG.values())
    print(f"Catalog in the system prompt: {len(catalog_lines.split())} words; "
          f"all {len(CATALOG)} skills in full: {all_bodies} words\n")

    for number, request in enumerate(REQUESTS, start=1):
        print(f"=== REQUEST {number}: {request[:60]}...")
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        if not any(m.type == "tool" for m in result["messages"]):
            print("  (no skill loaded)")
        answer_lines = result["messages"][-1].content.strip().splitlines()
        print("\n".join(f"  | {line}" for line in answer_lines[:8]))
        if len(answer_lines) > 8:
            print(f"  | ... ({len(answer_lines) - 8} more lines)")
        print()


if __name__ == "__main__":
    main()
