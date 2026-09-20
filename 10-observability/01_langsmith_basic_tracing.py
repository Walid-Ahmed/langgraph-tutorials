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
result = graph.invoke(
    {"team": "Zamalek SC in Egypt"},
config={"run_name": "Zamalek Facts"}
)

print(result["answer"])