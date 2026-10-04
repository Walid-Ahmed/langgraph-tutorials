# Research Assistant graph - a five-node sequential pipeline, built with no
# tracing code at all.
#
# This module defines the state, the five nodes (planner, document_reader,
# web_enricher, synthesizer, report_writer) and wires them in a straight line:
# START -> planner -> document_reader -> web_enricher -> synthesizer ->
# report_writer -> END. It exports the compiled graph as `graph` and is imported
# by run_traced.py; it is not meant to be run on its own.
#
# Why this matters:
# - Nothing here was written for LangSmith. With LANGSMITH_TRACING=true the
#   graph, each node, each ChatOpenAI call and each tool call still shows up as
#   a nested run, because all of them are Runnables.
# - planner and report_writer each nest a ChatOpenAI run; document_reader and
#   web_enricher each nest a tool run; synthesizer nests nothing, because a
#   string join is plain Python.
# - It is a fixed workflow, NOT an agent: the path never depends on the model.
#
# Before you use this:
# 1. OPENAI_API_KEY and SERPER_API_KEY must already be in the environment when
#    this module is imported (the models and the Serper wrapper are created at
#    import time). run_traced.py calls load_dotenv() first for this reason.

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from tools import search_local_docs, web_search

planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
writer_llm = ChatOpenAI(model="gpt-4o", temperature=0)


class ResearchState(TypedDict):
    query: str
    refined_query: str
    doc_sections: str
    web_results: str
    consolidated_context: str
    final_report: str
    session_id: str
    steps_taken: Annotated[list[str], operator.add]


def planner(state: ResearchState) -> dict:
    prompt = f"Rewrite this question to be specific and searchable: {state['query']}"
    refined = planner_llm.invoke(prompt).content
    return {"refined_query": refined, "steps_taken": ["planner"]}


def document_reader(state: ResearchState) -> dict:
    sections = search_local_docs.invoke({"query": state["refined_query"]})
    return {"doc_sections": sections, "steps_taken": ["document_reader"]}


def web_enricher(state: ResearchState) -> dict:
    results = web_search.invoke({"query": state["refined_query"]})
    return {"web_results": results, "steps_taken": ["web_enricher"]}


def synthesizer(state: ResearchState) -> dict:
    context = f"LOCAL SOURCES:\n{state['doc_sections']}\n\nWEB SOURCES:\n{state['web_results']}"
    return {"consolidated_context": context, "steps_taken": ["synthesizer"]}


def report_writer(state: ResearchState) -> dict:
    prompt = (f"Question: {state['refined_query']}\n\nContext:\n{state['consolidated_context']}\n\n"
              "Write a concise, well-organized answer.")
    report = writer_llm.invoke(prompt).content
    return {"final_report": report, "steps_taken": ["report_writer"]}


from langgraph.graph import StateGraph, START, END

builder = StateGraph(ResearchState)
for name, fn in [("planner", planner), ("document_reader", document_reader),
                  ("web_enricher", web_enricher), ("synthesizer", synthesizer),
                  ("report_writer", report_writer)]:
    builder.add_node(name, fn)

builder.add_edge(START, "planner")
builder.add_edge("planner", "document_reader")
builder.add_edge("document_reader", "web_enricher")
builder.add_edge("web_enricher", "synthesizer")
builder.add_edge("synthesizer", "report_writer")
builder.add_edge("report_writer", END)

graph = builder.compile()
