# 07 — Skills: Procedural Memory in Files

A **skill** is a written method for one kind of task (a checklist, a set of steps,
the format of the answer) kept in a `SKILL.md` file instead of in the agent's
system prompt. It's procedural memory, like `06_procedural_memory.py`, with two
differences: people write it, not an optimizer, and it's the same for every user,
versioned in git beside the code. (Book Chapter 11, section "Skills".)

A skill is not a tool. A tool *does* something (searches, sends, computes); a skill
tells the model *how* to do a task. An agent can use both.

## Files

| File | What it shows | LLM? |
|---|---|---:|
| [`01_review_without_skill.py`](01_review_without_skill.py) | the baseline: the review method written into the system prompt | yes |
| [`02_review_with_skill.py`](02_review_with_skill.py) | the same review, with the method loaded from `skills/paper_review/SKILL.md` by a `load_skill` node | yes |
| [`03_agent_loads_skill.py`](03_agent_loads_skill.py) | a ReAct agent that sees only a catalog of skill names and loads the one it needs with a `load_skill` tool | yes |
| [`skills/`](skills/) | three skills: `paper-review`, `literature-survey`, `proposal-evaluation` | — |

```text
07-skills/
├── 01_review_without_skill.py
├── 02_review_with_skill.py
├── 03_agent_loads_skill.py
└── skills/
    ├── paper_review/SKILL.md
    ├── literature_survey/SKILL.md
    └── proposal_evaluation/SKILL.md
```

Each `SKILL.md` starts with front matter, which is all the agent sees up front:

```markdown
---
name: paper-review
description: Review an academic research paper and recommend accept or reject. Use when asked to review, critique or referee a paper or abstract.
---
```

## Run

All three need `OPENAI_API_KEY` in the repository-root `.env`.

```bash
python "8-Long-Term-Memory/07-skills/01_review_without_skill.py"
python "8-Long-Term-Memory/07-skills/02_review_with_skill.py"
python "8-Long-Term-Memory/07-skills/03_agent_loads_skill.py"
```

Expected from `03` (answers vary):

```text
Catalog in the system prompt: 75 words; all 3 skills in full: 439 words

=== REQUEST 1: Please review this paper abstract: ...
  [skill] loaded paper-review
  | ## 1. Summary ...
=== REQUEST 2: Can you evaluate this grant proposal? ...
  [skill] loaded proposal-evaluation
  | | Criterion    | Score | Justification | ...
=== REQUEST 3: What does BM25 stand for? ...
  (no skill loaded)
  | BM25 stands for "Best Matching 25," ...
```

## Things worth noticing

- **State isn't context.** In `02`, the skill sits in `skill_instructions`, but the
  model sees it only because `review_paper` adds it to the messages.
- **Progressive disclosure.** In `03`, the catalog (names and descriptions) is
  always in the prompt, read in code at startup; a skill's full text enters the
  context only when the agent asks for it.
- **The model picks a name, never a path.** `load_skill` opens only names in the
  catalog, so a made-up name or `"../../.env"` gets an error message, not a file.
- **Signal plus guarantee.** gpt-4o-mini sometimes loaded the same skill twice.
  The tool result now says the skill is loaded (the signal), and the tool checks
  the conversation through `InjectedState` and won't send the same text twice
  (the guarantee).
- **Skills are trusted text.** Whatever is in a `SKILL.md` lands in the model's
  instructions. Load skills only from files you control and review.
