# A more elaborate prompt-chaining example: a 3-step draft -> reflect ->
# revise pipeline. An essay is drafted, an LLM reviewer critiques it, then
# a reviser rewrites the essay addressing every point of that feedback.

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

TOPIC = "Artificial Intelligence in Education"


# ---------------------------------------------------------
# 1. State
#
# Each field is written by one node and read by the next —
# the classic prompt chaining pattern.
# - topic          : set at invocation, never changes
# - draft          : written by the drafter, read by the reviewer
# - reflection     : written by the reviewer, read by the reviser
# - revised_essay  : final output written by the reviser
# ---------------------------------------------------------

class State(TypedDict):
    topic: str
    draft: str
    reflection: str
    revised_essay: str


# ---------------------------------------------------------
# 2. Node 1 — Drafter
#
# Writes a complete structured essay on the topic.
# ---------------------------------------------------------

DRAFTER_PROMPT = """You are an expert academic writer.

Write a complete, well-structured essay about the following topic:

Topic: {topic}

Requirements:
- Length: 800-1200 words
- Include a clear title
- Include:
  1. Introduction with a strong thesis statement
  2. 3-5 body paragraphs with evidence, examples, and analysis
  3. Conclusion that summarizes key points and offers insight
- Use formal academic English
- Make transitions between paragraphs smooth
- Avoid bullet points; write in full essay format
- Be original, coherent, and logically organized

Generate the complete essay now."""


def draft_essay(state: State):
    """Write the initial draft essay"""

    result = llm.invoke(DRAFTER_PROMPT.format(topic=state["topic"]))
    return {"draft": result.content}


# ---------------------------------------------------------
# 3. Node 2 — Reviewer
#
# Critically evaluates the draft across structure, clarity,
# argument strength, and writing style. Output feeds the reviser.
# ---------------------------------------------------------

REFLECTION_PROMPT = """You are an expert writing reviewer and editor.

Below is a draft essay:

{draft}

Your task is to critically evaluate this essay and provide constructive feedback.

Focus on the following dimensions:

1. Structure — Is the essay logically organized? Are transitions smooth?
2. Clarity — Are ideas clearly expressed? Any confusing sentences?
3. Strength of Argument — Is the thesis clear? Are claims well supported?
4. Writing Style — Is the writing concise and academically appropriate?

Instructions:
- Be critical but constructive.
- Identify both strengths and weaknesses.
- Provide specific examples from the essay when possible.
- Suggest concrete improvements.

Return your response in the following format:

## Overall Assessment
(2-4 sentences)

## Strengths
- ...

## Areas for Improvement
- ...

## Specific Revision Suggestions
- ...

## Final Score
Score: X/10
Brief justification."""


def reflect_on_essay(state: State):
    """Critically review the draft and produce structured feedback"""

    result = llm.invoke(REFLECTION_PROMPT.format(draft=state["draft"]))
    return {"reflection": result.content}


# ---------------------------------------------------------
# 4. Node 3 — Reviser
#
# Rewrites the draft addressing every point raised in the
# reflection. Receives both the original draft and the
# reviewer's feedback so it can target specific weaknesses.
# ---------------------------------------------------------

REVISION_PROMPT = """You are an expert academic writer and editor.

You are given two inputs:

## Original Draft
{original_draft}

## Critical Reflection
{reflection}

Your task is to revise the original essay by explicitly addressing every point of feedback.

Requirements:
- Strengthen weak arguments with better reasoning or examples
- Reorganize sections if structural issues were flagged
- Rewrite unclear sentences for clarity and precision
- Correct any style or grammar issues identified
- Preserve the core intent of the original essay

Return only the final revised essay."""


def revise_essay(state: State):
    """Revise the draft based on the reviewer's feedback"""

    result = llm.invoke(
        REVISION_PROMPT.format(
            original_draft=state["draft"],
            reflection=state["reflection"],
        )
    )
    return {"revised_essay": result.content}


# ---------------------------------------------------------
# 5. Build graph
#
# Linear chain: draft → reflect → revise
# No loops — each step runs exactly once.
# ---------------------------------------------------------

essay_builder = StateGraph(State)

essay_builder.add_node("draft_essay", draft_essay)
essay_builder.add_node("reflect_on_essay", reflect_on_essay)
essay_builder.add_node("revise_essay", revise_essay)

essay_builder.add_edge(START, "draft_essay")
essay_builder.add_edge("draft_essay", "reflect_on_essay")
essay_builder.add_edge("reflect_on_essay", "revise_essay")
essay_builder.add_edge("revise_essay", END)

essay_workflow = essay_builder.compile()

plot_graph(essay_workflow)

# ---------------------------------------------------------
# 6. Run
# ---------------------------------------------------------

state = essay_workflow.invoke({"topic": TOPIC})  # type: ignore[arg-type]

print("=" * 60)
print("INITIAL DRAFT")
print("=" * 60)
print(state["draft"])

print("\n" + "=" * 60)
print("REFLECTION")
print("=" * 60)
print(state["reflection"])

print("\n" + "=" * 60)
print("REVISED ESSAY")
print("=" * 60)
print(state["revised_essay"])
