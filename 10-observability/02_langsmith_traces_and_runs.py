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