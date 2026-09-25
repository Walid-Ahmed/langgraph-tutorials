# LangSmith traces and runs - the second observability example.
#
# This script builds a two-node LangGraph (get_facts -> summarize) in which the
# second node consumes the first node's output, then runs it once. Its job is to
# make the trace/run vocabulary concrete: one graph.invoke() is a single TRACE,
# and every operation inside it (each node, each LLM call) is a RUN nested in
# that trace's tree.
#
# Why this matters:
# - A trace is one logical execution; runs are the steps inside it. Seeing two
#   chained LLM calls under one trace is the clearest way to internalize this.
# - run_name names the top-level trace in LangSmith instead of the default
#   "LangGraph", so you can find this run in the dashboard.
# - As in file 01, tracing is automatic from the env vars; there is no tracing
#   code in the graph itself.
#
# Before you run this (same setup as 01_langsmith_basic_tracing.py):
# 1. Install this folder's dependencies:
#    pip install -r requirements.txt
# 2. Create a .env file in THIS folder (copy .env.example) containing:
#    OPENAI_API_KEY=your_openai_key
#    LANGSMITH_TRACING=true
#    LANGSMITH_API_KEY=your_langsmith_key      # from smith.langchain.com
#    LANGSMITH_PROJECT=10-observability
#
# Run it with (from the 10-observability folder, so load_dotenv finds .env):
#    python 02_langsmith_traces_and_runs.py
#
# Expected result: it prints a two-sentence summary, and a trace named
# "Zamalek Research1" appears at smith.langchain.com under "10-observability".
# Open that trace to see get_facts and summarize as two child runs. The text
# varies between runs (it is an LLM); what must hold is that get_facts feeds
# summarize.

from dotenv import load_dotenv
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load OpenAI and LangSmith configuration from .env.
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


# State shared by the graph nodes.
class State(TypedDict):
    team: str
    facts: str
    summary: str


# First node: ask the LLM for facts.
def get_facts(state: State):
    response = llm.invoke(
        f"Give me 3 facts about {state['team']}."
    )

    return {"facts": response.content}


# Second node: use the output of the first node
# to make another LLM call.
def summarize(state: State):
    response = llm.invoke(
        f"Summarize these facts in 2 sentences:\n{state['facts']}"
    )

    return {"summary": response.content}


# Build a two-node LangGraph.
builder = StateGraph(State)

builder.add_node("get_facts", get_facts)
builder.add_node("summarize", summarize)

# Execution order:
#
# START -> get_facts -> summarize -> END
builder.add_edge(START, "get_facts")
builder.add_edge("get_facts", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()


# One graph.invoke() represents one complete graph execution.
#
# In LangSmith, the complete execution is represented as a TRACE.
# Inside that trace, LangSmith records individual operations as RUNS.
#
# run_name gives the top-level execution a meaningful name
# in LangSmith instead of the default "LangGraph".
result = graph.invoke(
    {"team": "Zamalek SC in Egypt"},
    config={"run_name": "Zamalek Research1"}
)

print(result["summary"])
