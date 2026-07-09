# Demonstrates human-in-the-loop checkpointing with approval routing:
# an LLM creates a draft, the graph pauses before a review_decision node,
# the user reviews the actual draft outside the graph in the terminal,
# then update_state() saves that decision before invoke(None, config) resumes.

import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class ApprovalState(TypedDict):
    request: str
    draft: str
    approved: bool
    feedback: str
    final: str


def phase_banner(phase_num: int, title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  PHASE {phase_num}: {title}")
    print(f"{'=' * 55}")


def step_print(icon: str, label: str, detail: str = "") -> None:
    print(f"\n{icon} [{label}] {detail}")


def create_draft(state: ApprovalState) -> dict:
    """LLM creates the first draft from the user's request."""
    step_print("🤖", "LLM NODE", "Generating draft...")
    response = llm.invoke(f"Create a professional response for: {state['request']}")
    print(f"   Draft preview: {response.content[:120]}...")
    return {"draft": response.content}


def review_decision(state: ApprovalState) -> dict:
    """Process the decision the user made outside the graph during the pause."""
    step_print("👁️", "REVIEW DECISION NODE", "Using the user's saved decision.")
    decision = "APPROVED ✅" if state["approved"] else "REJECTED ❌"
    print(f"   Decision: {decision}")
    if state["feedback"]:
        print(f"   Feedback: {state['feedback']}")
    return {}


def route_human_decision(state: ApprovalState) -> Literal["finalize", "revise"]:
    """Conditional edge: human approval controls the next node."""
    if state["approved"]:
        print("\n→ Routing: APPROVE → finalize")
        return "finalize"
    print("\n→ Routing: REJECT → revise")
    return "revise"


def finalize(state: ApprovalState) -> dict:
    """Approved path: use the draft as the final answer."""
    step_print("✅", "FINALIZE NODE", "Draft approved — using as-is.")
    return {"final": state["draft"]}


def revise(state: ApprovalState) -> dict:
    """Rejected path: ask the LLM to revise using the human feedback."""
    step_print("✏️", "REVISE NODE", "Revising based on human feedback...")
    print(f"   Feedback: {state['feedback']}")
    response = llm.invoke(
        "Revise this draft based on the human feedback.\n\n"
        f"Draft:\n{state['draft']}\n\n"
        f"Feedback:\n{state['feedback']}"
    )
    print(f"   Revised draft preview: {response.content[:120]}...")
    return {"final": response.content}


def build_graph():
    builder = StateGraph(ApprovalState)

    builder.add_node("create_draft", create_draft)
    builder.add_node("review_decision", review_decision)
    builder.add_node("finalize", finalize)
    builder.add_node("revise", revise)

    builder.add_edge(START, "create_draft")
    builder.add_edge("create_draft", "review_decision")
    builder.add_conditional_edges(
        "review_decision",
        route_human_decision,
        {"finalize": "finalize", "revise": "revise"},
    )
    builder.add_edge("finalize", END)
    builder.add_edge("revise", END)

    checkpointer = MemorySaver()

    # `interrupt_before=["review_decision"]` means the graph stops after
    # `create_draft` and saves a checkpoint before the decision node runs.
    # The human can now inspect the real draft outside the graph, then
    # update the saved state with approval or feedback before resuming.
    return builder.compile(checkpointer=checkpointer, interrupt_before=["review_decision"])


def ask_for_request() -> str:
    """Collect the real task the user wants drafted."""
    request = input(
        "What should the assistant draft?\n"
        "Example: Write a thank-you email after a job interview.\n> "
    ).strip()
    if request:
        return request
    return "Write a thank-you email after a job interview"


def ask_for_review_decision(draft: str) -> dict:
    """Collect the real human decision from the terminal after showing the draft."""
    print("\nDraft to review:")
    print("-" * 55)
    print(draft)
    print("-" * 55)

    while True:
        decision = input("\nApprove this draft? (y/n): ").strip().lower()
        if decision in {"y", "yes"}:
            return {"approved": True, "feedback": ""}
        if decision in {"n", "no"}:
            feedback = input("What should be improved? ").strip()
            if not feedback:
                feedback = "Make the draft clearer and more polished."
            return {"approved": False, "feedback": feedback}
        print("Please type 'y' to approve or 'n' to request changes.")


def main() -> None:
    graph = build_graph()
    plot_graph(graph, "7-Checkpointing/diagrams/07_human_review_approval_graph.png")

    config: RunnableConfig = {"configurable": {"thread_id": "hitl-routing-demo"}}

    print("\n" + "=" * 55)
    print("  HUMAN-IN-THE-LOOP: TERMINAL REVIEW")
    print("=" * 55)
    print("\nGraph:")
    print("START → create_draft → ⏸ review_decision ──approve──→ finalize → END")
    print("                                         ╚──reject──→ revise   → END")

    phase_banner(1, "USER ENTERS A REAL DRAFTING REQUEST")
    user_request = ask_for_request()

    phase_banner(2, "RUN UNTIL INTERRUPT")
    result = graph.invoke(
        {
            "request": user_request,
            "draft": "",
            "approved": False,
            "feedback": "",
            "final": "",
        },
        config,
    )

    step_print("⏸️", "PAUSED", "Graph frozen before the review decision node.")

    phase_banner(3, "USER REVIEWS THE ACTUAL DRAFT")
    decision_update = ask_for_review_decision(result["draft"])

    # update_state() edits the saved checkpoint for this thread. The next
    # invoke(None, config) resumes from that checkpoint with the user's values.
    graph.update_state(config, decision_update)
    print(f"   Saved approved: {decision_update['approved']}")
    print(f"   Saved feedback: {decision_update['feedback']!r}")

    phase_banner(4, "RESUME FROM CHECKPOINT")
    print("   Resuming graph with invoke(None, config)...")
    final_state = graph.invoke(None, config)

    routed_to = "finalize" if decision_update["approved"] else "revise"
    step_print("🔀", "ROUTED TO", f"{routed_to} node")
    print("\nFinal answer:")
    print(final_state["final"])

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print("  interrupt_before pauses the graph after the draft is created")
    print("  the user reviews the actual draft in the terminal")
    print("  update_state saves the user's decision into the checkpoint")
    print("  invoke(None, config) resumes from that checkpoint")


if __name__ == "__main__":
    main()
