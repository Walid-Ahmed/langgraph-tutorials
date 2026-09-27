from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Single .env for this package (complete file: keys + LANGSMITH_PROJECT).
load_dotenv(Path(__file__).resolve().parent / ".env")

from .state import AgentState
from .nodes import planner, document_reader, web_enricher, synthesizer, report_writer


def build_graph():
    """
    Sequential document workflow (not a tool-calling agent).

    Flow: planner → document_reader → web_enricher → synthesizer → report_writer → END

    All steps are automatically traced to LangSmith when LANGSMITH_TRACING=true.
    No explicit tracing code needed — LangGraph sends traces via env vars.
    """
    g = StateGraph(AgentState)

    g.add_node("planner",         planner)
    g.add_node("document_reader", document_reader)
    g.add_node("web_enricher",    web_enricher)
    g.add_node("synthesizer",     synthesizer)
    g.add_node("report_writer",   report_writer)

    g.add_edge(START, "planner")
    g.add_edge("planner",         "document_reader")
    g.add_edge("document_reader", "web_enricher")
    g.add_edge("web_enricher",    "synthesizer")
    g.add_edge("synthesizer",     "report_writer")
    g.add_edge("report_writer",   END)

    return g.compile()


graph = build_graph()
