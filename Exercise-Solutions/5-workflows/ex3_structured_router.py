# Run from the repository root:
#   python "Exercise-Solutions/5-workflows/ex3_structured_router.py"
#
# Exercise 5.4 — A structured-output router (Appendix B, Listing B.14)
#
# What it shows: classify with a schema, then route on the label.
# The Literal type makes an out-of-list label impossible, so the router can
# never name a node that does not exist; the label IS the node name.
# Each category gets its own answer node with a category-specific prompt.
# Needs OPENAI_API_KEY in the repository-root .env file.
#
# Expected output (representative):
#   'What year did the Berlin Wall fall?'        -> factual
#   'Write a limerick about Mondays.'            -> creative
#   'Why does my Python loop skip items?'        -> technical

import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

sys.path.append(str(Path(__file__).resolve().parents[2]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class Route(BaseModel):
    category: Literal["factual", "creative", "technical"] = Field(
        description="factual = look-up questions; creative = writing requests; "
                    "technical = programming or engineering questions")


router_llm = llm.with_structured_output(Route)


class State(TypedDict):
    question: str
    category: str
    answer: str


# One instruction per category; make_answer_node turns each into a node.
ANSWER_PROMPTS = {
    "factual": "Answer this factual question concisely and accurately:",
    "creative": "Answer this request imaginatively and creatively:",
    "technical": "Answer this technical question with precise detail and, if helpful, code:",
}


def make_answer_node(name: str):
    def answer(state: State) -> dict:
        reply = llm.invoke(f"{ANSWER_PROMPTS[name]}\n\n{state['question']}")
        return {"answer": reply.content}
    return answer


def classify(state: State) -> dict:
    return {"category": router_llm.invoke(state["question"]).category}


def route(state: State) -> str:
    return state["category"]                 # the label IS the node name


def main() -> None:
    builder = StateGraph(State)
    builder.add_node("classify", classify)
    for name in ("factual", "creative", "technical"):
        builder.add_node(name, make_answer_node(name))    # category-specific prompt
        builder.add_edge(name, END)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route, ["factual", "creative", "technical"])

    app = builder.compile()
    plot_graph(app, print_mermaid=False)

    for question in ["What year did the Berlin Wall fall?",
                     "Write a limerick about Mondays.",
                     "Why does my Python loop skip items?"]:
        result = app.invoke({"question": question})
        print(f"{repr(question):<44} -> {result['category']}")


if __name__ == "__main__":
    main()
