# Run from the repository root:
#   python "5-Workflows/03_parallelization_creative.py"
#
# What it does:
#   Builds a study pack for one topic. Three writers run in parallel from the
#   same topic, a short summary, a three-question quiz, and an everyday
#   analogy, and an assemble node joins them into one pack.
#
# What it demonstrates (Chapter 5 — parallelization):
#   - static fan-out: three edges out of START, known when you build the graph
#   - fan-in: one node with three incoming edges runs after all three finish
#   - each writer owns one state field, so no reducer is needed
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (wording varies):
#   === STUDY PACK: Python generators ===
#   SUMMARY ... QUIZ (3 questions) ... ANALOGY ...

import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


class State(TypedDict):
    topic: str
    summary: str      # one writer per field: no reducer needed
    quiz: str
    analogy: str
    study_pack: str


def write_summary(state: State) -> dict:
    return {"summary": llm.invoke(f"Explain {state['topic']} to a beginner in 3 sentences.").text}


def write_quiz(state: State) -> dict:
    return {"quiz": llm.invoke(f"Write 3 short quiz questions, with answers, on {state['topic']}.").text}


def write_analogy(state: State) -> dict:
    return {"analogy": llm.invoke(f"Give one everyday analogy for {state['topic']}, in 2 sentences.").text}


def assemble_pack(state: State) -> dict:
    """Runs once, after all three writers: the fan-in."""
    pack = (f"=== STUDY PACK: {state['topic']} ===\n\nSUMMARY\n{state['summary']}\n\n"
            f"QUIZ\n{state['quiz']}\n\nANALOGY\n{state['analogy']}")
    return {"study_pack": pack}


builder = StateGraph(State)
for name, writer in [("write_summary", write_summary), ("write_quiz", write_quiz),
                     ("write_analogy", write_analogy)]:
    builder.add_node(name, writer)
    builder.add_edge(START, name)              # fan-out
    builder.add_edge(name, "assemble_pack")    # fan-in
builder.add_node("assemble_pack", assemble_pack)
builder.add_edge("assemble_pack", END)
graph = builder.compile()


def main() -> None:
    plot_graph(graph, str(Path(__file__).resolve().parent / "diagrams" / "03_parallelization_creative_graph.png"),
               print_mermaid=False)
    result = graph.invoke({"topic": "Python generators"})  # type: ignore[arg-type]
    print(result["study_pack"])


if __name__ == "__main__":
    main()
