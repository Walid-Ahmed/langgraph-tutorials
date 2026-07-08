# A realistic checkpointed workflow: intake -> analyze -> (revise -> analyze)*
# -> finalize, where an LLM scores a document and loops it through revisions
# until the quality score is high enough or MAX_ITERATIONS is hit. Each node
# is its own checkpoint, and the full checkpoint history is printed at the end.

import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

MAX_ITERATIONS = 3  # hard stop so a stubborn low score can't loop forever


# ---------------------------------------------------------
# 1. State
#
# `iterations` tracks how many analyze/revise passes have run,
# so the router can cut the loop off even if the LLM never
# scores the document high enough on its own.
# ---------------------------------------------------------

class DocumentState(TypedDict):
    document_title: str
    document_content: str
    processing_stage: str
    quality_score: int
    issues_found: list[str]
    revisions_made: list[str]
    approved: bool
    iterations: int


# ---------------------------------------------------------
# 2. Structured output schema
#
# Forces the analysis LLM call to return a typed score/issues/
# recommendation instead of freeform text we'd have to parse.
# ---------------------------------------------------------

class QualityAnalysis(BaseModel):
    score: int = Field(description="Quality score 1-10", ge=1, le=10)
    issues: list[str] = Field(description="List of issues found")
    recommendation: Literal["approve", "revise", "reject"]


# ---------------------------------------------------------
# 3. Nodes
#
# Each node is its own super-step, so each one produces its
# own checkpoint — that's what the history print at the bottom
# of this file makes visible.
# ---------------------------------------------------------

def intake_document(state: DocumentState):
    """Receive and log the document. No LLM call — just marks progress."""
    print(f"\n📄 INTAKE: Processing '{state['document_title']}'")
    print(f"   Content length: {len(state['document_content'])} characters")

    return {"processing_stage": "intake_complete"}


def analyze_quality(state: DocumentState):
    """Score the document and bump the iteration count on every pass,
    whether the verdict is approve or revise."""
    print("\n🔍 ANALYSIS: Evaluating quality...")

    analyzer_llm = llm.with_structured_output(QualityAnalysis)

    prompt = f"""Analyze this document for quality:

        Title: {state['document_title']}
        Content: {state['document_content']}

        Evaluate for:
        - Clarity and structure
        - Grammar and spelling
        - Completeness
        - Professional tone

        Provide a score (1-10) and list specific issues."""

    analysis = cast(QualityAnalysis, analyzer_llm.invoke(prompt))

    print(f"   Quality Score: {analysis.score}/10")
    print(f"   Issues Found: {len(analysis.issues)}")

    return {
        "processing_stage": "analysis_complete",
        "quality_score": analysis.score,
        "issues_found": analysis.issues,
        "approved": analysis.recommendation == "approve",
        "iterations": state.get("iterations", 0) + 1,
    }


def revise_document(state: DocumentState):
    """Rewrite the document to address the issues from the last analysis pass."""
    print("\n✏️  REVISION: Improving document...")

    issues_text = "\n".join(f"- {issue}" for issue in state["issues_found"])

    prompt = f"""Revise this document to address these issues:

        Original:
        {state['document_content']}

        Issues to fix:
        {issues_text}

        Provide an improved version (keep it concise, around same length)."""

    revised_content = llm.invoke(prompt).content

    print(f"   Revisions applied: {len(state['issues_found'])} issues addressed")

    return {
        "processing_stage": "revision_complete",
        "document_content": revised_content,
        "revisions_made": state["issues_found"],
    }


def finalize_document(state: DocumentState):
    """Mark the document as complete, whether it earned the score or hit the cap."""
    print("\n✅ FINALIZED: Document approved and ready")

    return {"processing_stage": "finalized", "approved": True}


# ---------------------------------------------------------
# 4. Conditional edge (router)
#
# Three ways out of the analyze/revise loop:
#   - score is high enough        → finalize
#   - MAX_ITERATIONS reached      → finalize anyway (avoid an infinite loop)
#   - otherwise                   → revise and re-analyze
# ---------------------------------------------------------

def route_after_analysis(state: DocumentState) -> Literal["revise", "finalize"]:
    if state["quality_score"] >= 8:
        print("\n→ Routing to finalize (high quality)")
        return "finalize"
    if state["iterations"] >= MAX_ITERATIONS:
        print(f"\n→ Routing to finalize (hit MAX_ITERATIONS={MAX_ITERATIONS})")
        return "finalize"
    print("\n→ Routing to revise (needs improvement)")
    return "revise"


# ---------------------------------------------------------
# 5. Build graph
# ---------------------------------------------------------

builder = StateGraph(DocumentState)

builder.add_node("intake", intake_document)
builder.add_node("analyze", analyze_quality)
builder.add_node("revise", revise_document)
builder.add_node("finalize", finalize_document)

builder.add_edge(START, "intake")
builder.add_edge("intake", "analyze")
builder.add_conditional_edges(
    "analyze",
    route_after_analysis,
    {"revise": "revise", "finalize": "finalize"},
)
builder.add_edge("revise", "analyze")  # loop back for re-analysis
builder.add_edge("finalize", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

plot_graph(graph)


# ---------------------------------------------------------
# 6. Run once on a deliberately weak document so the loop
#    actually executes at least one revise/analyze cycle.
# ---------------------------------------------------------

config: RunnableConfig = {"configurable": {"thread_id": "doc_review_001"}}

initial_input: DocumentState = {
    "document_title": "Q4 Sales Report",
    "document_content": "Sales went up this quarter. We did good. Revenue increased.",
    "processing_stage": "",
    "quality_score": 0,
    "issues_found": [],
    "revisions_made": [],
    "approved": False,
    "iterations": 0,
}

result = graph.invoke(initial_input, config=config)


# ---------------------------------------------------------
# 7. Inspect the full checkpoint history
#
# Every node run above (intake, each analyze, each revise,
# finalize) saved its own checkpoint. get_state_history returns
# all of them for this thread, newest first.
# ---------------------------------------------------------

def print_snapshot_data(snapshot_state) -> None:
    thread_id = snapshot_state.config["configurable"]["thread_id"]
    print(f"Thread ID: {thread_id}")

    checkpoint_id = snapshot_state.config["configurable"]["checkpoint_id"]
    print(f"Checkpoint ID: {checkpoint_id}")

    next_nodes = snapshot_state.next if snapshot_state.next else "None (workflow complete)"
    print(f"Next Nodes -> {next_nodes}")

    print("\nState Values:")
    print(f"  Title: {snapshot_state.values.get('document_title', 'N/A')}")
    print(f"  Stage: {snapshot_state.values.get('processing_stage', 'N/A')}")
    print(f"  Quality Score: {snapshot_state.values.get('quality_score', 0)}")
    print(f"  Approved: {snapshot_state.values.get('approved', False)}")
    print(f"  Issues Found: {snapshot_state.values.get('issues_found', 'None')}")


print("-" * 50)
print("Full State History")
print("-" * 50)

history = list(graph.get_state_history(config))
total_checkpoints = len(history)
print(f"Total Checkpoints: {total_checkpoints}")

print("\nCheckpoint Timeline (newest to oldest)")
print("=" * 50)

for i, checkpoint in enumerate(history):
    print(f"\n{'*' * 30}")
    print(f"Checkpoint Snapshot {total_checkpoints - i}")
    print(f"{'*' * 30}")
    print_snapshot_data(checkpoint)
