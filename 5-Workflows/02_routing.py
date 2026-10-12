# Run from the repository root:
#   python "5-Workflows/02_routing.py"
#
# What it does:
#   Routes a student's question to the right tutor. A classifier node asks the
#   model which subject the question belongs to (math, code or writing) and
#   writes that label into state; a plain-Python router reads the label and
#   sends the question to the matching tutor node, which answers in that
#   subject's style.
#
# What it demonstrates (Chapter 5 — routing):
#   - structured output with a Literal: the model can only answer one of three labels
#   - the two-piece split: the model call lives in a NODE (classify_question),
#     the branching lives in a ROUTER (route_to_tutor) that only reads state
#   - each route gets its own prompt; that's why the categories deserve separate nodes
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (answers vary; the labels should match):
#   Q: What's the derivative of x^2 * sin(x)? -> math
#   Q: Why does my Python loop print the last value three times? -> code
#   Q: How do I make my essay's opening paragraph stronger? -> writing

import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class State(TypedDict):
    question: str
    subject: str      # written by classify_question, read by route_to_tutor
    answer: str


class TutorChoice(BaseModel):
    subject: Literal["math", "code", "writing"] = Field(
        description="math = calculation or proof; code = programming; writing = essays, grammar, style")


classifier = llm.with_structured_output(TutorChoice)


def classify_question(state: State) -> dict:          # a NODE: makes the call, writes the label
    choice = cast(TutorChoice, classifier.invoke([
        SystemMessage("Decide which tutor should answer this student's question."),
        HumanMessage(state["question"]),
    ]))
    return {"subject": choice.subject}


def route_to_tutor(state: State) -> str:              # a ROUTER: pure Python, reads the label
    return state["subject"]


TUTOR_STYLE = {
    "math": "You are a math tutor. Show each step, then check the result.",
    "code": "You are a programming tutor. Explain the cause, then show a fixed snippet.",
    "writing": "You are a writing tutor. Give two concrete suggestions with a before/after example.",
}


def make_tutor(subject: str):
    """One tutor node per subject: same shape, its own prompt."""
    def tutor(state: State) -> dict:
        reply = llm.invoke([SystemMessage(TUTOR_STYLE[subject]), HumanMessage(state["question"])])
        return {"answer": reply.text}
    return tutor


builder = StateGraph(State)
builder.add_node("classify_question", classify_question)
for subject in TUTOR_STYLE:
    builder.add_node(f"{subject}_tutor", make_tutor(subject))
    builder.add_edge(f"{subject}_tutor", END)
builder.add_edge(START, "classify_question")
builder.add_conditional_edges("classify_question", route_to_tutor,
                              {"math": "math_tutor", "code": "code_tutor", "writing": "writing_tutor"})
graph = builder.compile()

QUESTIONS = [
    "What's the derivative of x^2 * sin(x)?",
    "Why does my Python loop print the last value three times?",
    "How do I make my essay's opening paragraph stronger?",
]


def main() -> None:
    plot_graph(graph, str(Path(__file__).resolve().parent / "diagrams" / "02_routing_graph.png"),
               print_mermaid=False)
    for question in QUESTIONS:
        result = graph.invoke({"question": question})  # type: ignore[arg-type]
        print(f"\nQ: {question} -> {result['subject']}")
        print("   " + result["answer"].strip().splitlines()[0][:110] + " ...")


if __name__ == "__main__":
    main()
