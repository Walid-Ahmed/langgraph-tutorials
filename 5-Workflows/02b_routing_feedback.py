# Run from the repository root:
#   python "5-Workflows/02b_routing_feedback.py"
#
# What it does:
#   Triages customer feedback. A classifier node asks the LLM to label each
#   piece of feedback "positive", "negative", or "neutral" (structured output),
#   and a conditional edge dispatches it to the matching reply node.
#
# What it demonstrates (Chapter 5, §5.5 — Routing):
#   - with_structured_output + a Literal schema, so the label is guaranteed
#     to be one of three values and never free text
#   - the node/router split: the LLM call lives in a node and writes its
#     label into state; a plain-Python router reads that label
#   - a complete routing graph: START -> classify -> one of three -> END
#
# Prerequisites:
#   - pip install -r requirements.txt   (langgraph, langchain-openai, python-dotenv)
#   - OPENAI_API_KEY in the repo-root .env file (Appendix A)
#
# Expected output (labels are stable at temperature=0; see the book for a full run):
#   Great service, the order arrived a day early!
#     -> positive: Thank you for the kind words! ...
#   The app keeps crashing when I try to pay.
#     -> negative: We're sorry to hear that. ...
#   I changed my delivery address last week.
#     -> neutral: Thanks for your feedback. ...

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()


# 1. The structured-output schema: the "form" the classifier must fill in.
#    Literal makes any label outside these three impossible.
class FeedbackCategory(BaseModel):
    category: Literal["positive", "negative", "neutral"] = Field(
        description="The category of the customer feedback"
    )


# 2. The baton: the feedback coming in, the label, and the reply going out.
class State(TypedDict):
    feedback: str
    category: str       # written by classify_feedback, read by route
    response: str       # written by exactly one specialist node


# 3. temperature=0 so the same feedback gets the same label run after run.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(FeedbackCategory)


# 4. The classifier NODE: the only LLM call in the graph. Its judgment is
#    written into state, so it can be logged, inspected, and tested.
def classify_feedback(state: State) -> dict:
    result = structured_llm.invoke([
        HumanMessage(content=f"Classify this customer feedback: {state['feedback']}")
    ])
    return {"category": result.category}


# 5. The specialist nodes: one per category. Each returns only its own field.
def positive_node(state: State) -> dict:
    return {"response": "Thank you for the kind words! We're happy you had a good experience. 😊"}


def negative_node(state: State) -> dict:
    return {"response": "We're sorry to hear that. Our support team will look into this issue right away."}


def neutral_node(state: State) -> dict:
    return {"response": "Thanks for your feedback. We'll take it into consideration."}


# 6. The ROUTER: pure Python, no LLM. It reads the label and names the next node.
#    The Literal return type also tells LangGraph every possible destination.
def route(state: State) -> Literal["positive_node", "negative_node", "neutral_node"]:
    if state["category"] == "positive":
        return "positive_node"
    elif state["category"] == "negative":
        return "negative_node"
    else:
        return "neutral_node"


# 7. The wiring: one way in, a three-way fork, three ways out.
builder = StateGraph(State)

builder.add_node("classify_feedback", classify_feedback)
builder.add_node("positive_node", positive_node)
builder.add_node("negative_node", negative_node)
builder.add_node("neutral_node", neutral_node)

builder.add_edge(START, "classify_feedback")
builder.add_conditional_edges(
    "classify_feedback",
    route,
    {
        "positive_node": "positive_node",
        "negative_node": "negative_node",
        "neutral_node": "neutral_node",
    },
)

# Every branch needs its own exit (Chapter 4's classic wiring bug).
builder.add_edge("positive_node", END)
builder.add_edge("negative_node", END)
builder.add_edge("neutral_node", END)

graph = builder.compile()


if __name__ == "__main__":
    samples = [
        "Great service, the order arrived a day early!",
        "The app keeps crashing when I try to pay.",
        "I changed my delivery address last week.",
    ]
    for text in samples:
        result = graph.invoke({"feedback": text})
        print(text)
        print(f"  -> {result['category']}: {result['response']}")
