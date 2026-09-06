"""
Sequential Document Intelligence Agent — nodes.
All LangChain / LangGraph calls are auto-traced to LangSmith via env vars.
No explicit tracing code needed in these nodes.
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
import os

from .state import AgentState
from .tools import search_document

# ── LLM clients ──────────────────────────────────────────────────────────
# Fast/cheap model for lightweight steps (question rewriting, synthesis)
_fast = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Stronger model for the final formatting/writing pass
_writer = ChatOpenAI(
    model="gpt-5.4",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
)

_serper = GoogleSerperAPIWrapper()

# ── Nodes ─────────────────────────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    """Rewrites the user question for clarity and precision."""
    response = _fast.invoke([
        SystemMessage(content=(
            "You are a research question refiner. "
            "Rewrite the user question to be more specific and searchable. "
            "Return ONLY the rewritten question, nothing else."
        )),
        HumanMessage(content=state["question"]),
    ])
    return {
        "refined_question": response.content.strip(),
        "steps_taken": state.get("steps_taken", []) + ["planner"],
    }


def document_reader(state: AgentState) -> dict:
    """Searches the local knowledge-base document for relevant sections."""
    sections = search_document(state["refined_question"], top_k=3)
    return {
        "doc_sections": sections,
        "steps_taken": state.get("steps_taken", []) + ["document_reader"],
    }


def web_enricher(state: AgentState) -> dict:
    """Fetches the latest information from the web using Google Serper."""
    try:
        web_text = _serper.run(state["refined_question"])
    except Exception as exc:
        web_text = f"[Web search unavailable: {exc}]"
    return {
        "web_results": web_text,
        "steps_taken": state.get("steps_taken", []) + ["web_enricher"],
    }


def synthesizer(state: AgentState) -> dict:
    """Combines document knowledge and web results into a coherent analysis."""
    doc_context = "\n\n---\n\n".join(state["doc_sections"])
    synthesis = _fast.invoke([
        SystemMessage(content=(
            "You are a research synthesizer. Given knowledge from a document and "
            "from the web, combine both into a clear, structured analysis. "
            "Cite sources where possible. Use markdown formatting."
        )),
        HumanMessage(content=(
            f"Question: {state['refined_question']}\n\n"
            f"=== DOCUMENT KNOWLEDGE ===\n{doc_context}\n\n"
            f"=== WEB SEARCH RESULTS ===\n{state['web_results']}"
        )),
    ])
    return {
        "synthesis": synthesis.content,
        "steps_taken": state.get("steps_taken", []) + ["synthesizer"],
    }


def report_writer(state: AgentState) -> dict:
    """Formats the synthesis into a polished final report."""
    try:
        report = _writer.invoke([
            SystemMessage(content=(
                "You are a technical report writer. Format the given analysis into "
                "a clean, well-structured report with: a one-sentence TL;DR at the top, "
                "key findings as bullet points, and a brief conclusion. "
                "Keep it under 400 words."
            )),
            HumanMessage(content=state["synthesis"]),
        ])
        final = report.content
    except Exception:
        # Fallback to raw synthesis if the writer model is unavailable
        final = state["synthesis"]

    return {
        "final_report": final,
        "steps_taken": state.get("steps_taken", []) + ["report_writer"],
    }