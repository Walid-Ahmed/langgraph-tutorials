# LangSmith basic tracing - the first observability example in the sequence.
#
# This script builds the smallest possible LangGraph (one node that asks an LLM
# for a brief overview of a topic) and runs it once. It shows that LangSmith
# tracing happens automatically: with the three LANGSMITH_* environment
# variables set, the graph run and its nested LLM call are recorded in the
# LangSmith dashboard without a single line of tracing code.
#
# Why this matters:
# - Tracing is "free": you turn it on with env vars, not by instrumenting code.
# - One graph.invoke() shows up in LangSmith as ONE trace; run_name gives that
#   trace a readable title instead of the default "LangGraph".
# - It deliberately does NOT use @traceable, tags, or metadata yet - those come
#   in the later files. This is the bare minimum that produces a trace.
#
# Before you run this:
# 1. Install this folder's dependencies:
#    pip install -r requirements.txt
# 2. Create a .env file in THIS folder (copy .env.example) containing:
#    OPENAI_API_KEY=your_openai_key
#    LANGSMITH_TRACING=true
#    LANGSMITH_API_KEY=your_langsmith_key      # from smith.langchain.com
#    LANGSMITH_PROJECT=10-observability
#
# Run it with (from the 10-observability folder, so load_dotenv finds .env):
#    python 01_langsmith_basic_tracing.py
#
# Expected result: it prints a short overview paragraph of the team, and a trace
# named "Zamalek Facts" appears at smith.langchain.com under the
# "10-observability" project. The exact text varies (it is an LLM); what must
# hold is that it describes the team you passed in.
#
# Then run the next file in the sequence:
#    python 02_langsmith_traces_and_runs.py

from dotenv import load_dotenv
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env.
# LangSmith tracing is enabled through:
#   LANGSMITH_TRACING=true
#   LANGSMITH_API_KEY=...
#   LANGSMITH_PROJECT=10-observability
load_dotenv()

# Create the LLM used by the graph.
llm = ChatOpenAI(model="gpt-4o-mini")


# Define the state that flows through the graph.
class State(TypedDict):
    team: str
    answer: str


# Define a graph node.
# The node receives the current state and returns a state update.
def answer_question(state: State):
    response = llm.invoke(
        f"Give me a brief overview of {state['team']}."
    )

    return {"answer": response.content}


# Build a simple LangGraph with one node.
builder = StateGraph(State)

builder.add_node("answer_question", answer_question)

builder.add_edge(START, "answer_question")
builder.add_edge("answer_question", END)

graph = builder.compile()


# Invoke the graph normally.
#
# No LangSmith-specific tracing code is required here.
# Because LANGSMITH_TRACING=true, LangSmith automatically
# records the graph execution and its nested operations.
#
# run_name names the top-level trace in LangSmith.
result = graph.invoke(
    {"team": "Zamalek SC in Egypt"},
    config={"run_name": "Zamalek Facts"},
)

print(result["answer"])
