# Run from the repository root:
#   python "8-Long-Term-Memory/06_procedural_memory.py"
#
# What it does:
#   Keeps the assistant's instructions as named sections in the Store. Before
#   every answer the system prompt is rebuilt from those sections with exact
#   get() calls. When the user gives feedback, a separate LangMem prompt
#   optimizer rewrites only the sections the feedback is about, and they are
#   saved back with put(). The next answer follows the new instructions.
#
# What it demonstrates (Chapter 11 — procedural memory):
#   - procedural memory = the instructions that shape behavior
#   - exact-key reads (no embedding search) for stable, named sections
#   - seeding defaults only when missing, so learned instructions survive restarts
#   - an optimizer that updates one section and leaves the others alone
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env; langmem.
#
# Expected output (wording varies):
#   Before   -> a long answer with no practice question
#   Updated  -> ['answer-format'] (the tone section is unchanged)
#   After    -> a short answer that ends with one practice question
#   Other user -> still has the default answer-format section

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from langmem import create_multi_prompt_optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

store = InMemoryStore()  # no index: sections are read by exact key
model = init_chat_model("openai:gpt-4o-mini", temperature=0)
optimizer = create_multi_prompt_optimizer("openai:gpt-4o-mini", kind="prompt_memory")

DEFAULT_SECTIONS = {
    "tone": "Be friendly and encouraging.",
    "answer_format": "Explain the concept thoroughly with background and examples.",
}

# when_to_update tells the optimizer which feedback each section is responsible for.
SECTION_SPECS = {
    "tone": {"key": "tone",
             "when_to_update": "Update when feedback is about tone or attitude."},
    "answer-format": {"key": "answer_format",
                      "when_to_update": "Update when feedback is about length, structure or what an answer must include."},
}


def instructions_namespace(user_id: str) -> tuple[str, str, str]:
    return ("assistant", user_id, "instructions")


def seed_defaults(user_id: str) -> None:
    """Write defaults only if missing: never overwrite learned instructions."""
    for key, text in DEFAULT_SECTIONS.items():
        if store.get(instructions_namespace(user_id), key) is None:
            store.put(instructions_namespace(user_id), key, {"prompt": text, "source": "default"})


def read_section(user_id: str, key: str) -> str:
    return store.get(instructions_namespace(user_id), key).value["prompt"]


def build_system_prompt(user_id: str) -> str:
    """Rebuilt on every call from the latest stored sections; never stored whole."""
    seed_defaults(user_id)
    return (
        "You are a study assistant.\n"
        f"Tone: {read_section(user_id, 'tone')}\n"
        f"Answer format: {read_section(user_id, 'answer_format')}"
    )


def ask(user_id: str, question: str) -> str:
    reply = model.invoke([{"role": "system", "content": build_system_prompt(user_id)},
                          {"role": "user", "content": question}])
    return reply.content


def learn_from_feedback(user_id: str, question: str, answer: str, feedback: str) -> list[str]:
    """One optimizer call proposes new sections; save only those that changed."""
    prompts = [{"name": name,
                "prompt": read_section(user_id, spec["key"]),
                "update_instructions": "Keep each instruction short and specific.",
                "when_to_update": spec["when_to_update"]}
               for name, spec in SECTION_SPECS.items()]
    trajectory = [HumanMessage(question), AIMessage(answer)]
    proposed = optimizer.invoke({"trajectories": [(trajectory, feedback)], "prompts": prompts})

    updated = []
    for old, new in zip(prompts, proposed):
        if new["prompt"] != old["prompt"]:
            key = SECTION_SPECS[old["name"]]["key"]
            store.put(instructions_namespace(user_id), key,
                      {"prompt": new["prompt"], "source": "optimized_from_feedback"})
            updated.append(old["name"])
    return updated


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repository-root .env file.")

    user = "maya"
    question = "What is a Python generator?"

    print("=== BEFORE FEEDBACK ===")
    print(build_system_prompt(user))
    before = ask(user, question)
    print(f"\n{before}\n({len(before.split())} words)")

    feedback = "Too long. Keep answers under 100 words and end with one practice question."
    print(f"\n=== FEEDBACK: {feedback}")
    print("Updated sections:", learn_from_feedback(user, question, before, feedback) or ["none"])

    print("\n=== AFTER FEEDBACK ===")
    print(build_system_prompt(user))
    after = ask(user, "What is a Python decorator?")
    print(f"\n{after}\n({len(after.split())} words)")

    print("\n=== DIFFERENT USER — still the default instructions ===")
    print(build_system_prompt("another-user"))


if __name__ == "__main__":
    main()
