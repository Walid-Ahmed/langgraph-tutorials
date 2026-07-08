# Orchestrator-workers variant: an orchestrator plans a report's sections
# via structured output, dynamically dispatches one worker per section with
# Send(), and a synthesizer stitches the completed sections into one
# markdown report.

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()


# ---------------------------------------------------------
# 1. Structured Output Schemas
#
# The orchestrator uses structured output to create a report plan.
# Each Section becomes one worker task.
# ---------------------------------------------------------
class Section(BaseModel):
    """One section in the report plan."""

    name: str = Field(description="Name of this report section")
    description: str = Field(description="What this section should cover")


class Sections(BaseModel):
    """Complete report plan."""

    sections: list[Section] = Field(
        description="Report sections to write",
        min_length=3,
        max_length=5,
    )


# ---------------------------------------------------------
# 2. State Definitions
#
# State is the shared graph state.
# WorkerState is the smaller state sent to each worker.
#
# completed_sections uses a reducer because many workers write
# to this key in parallel.
# ---------------------------------------------------------
class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list[str], operator.add]
    final_report: str


class WorkerState(TypedDict):
    section: Section


llm = ChatOpenAI(model="gpt-4o", temperature=0)
planner = llm.with_structured_output(Sections)


# ---------------------------------------------------------
# 3. Orchestrator Node
#
# The orchestrator creates the plan, but does not write the
# report sections itself.
# ---------------------------------------------------------
def orchestrator(state: State) -> dict:
    """Generate a section plan for the report."""
    report_sections = planner.invoke(
        [
            SystemMessage(
                content=(
                    "Generate a concise report plan with 3-5 sections. "
                    "Each section should have a clear name and description."
                )
            ),
            HumanMessage(content=f"Here is the report topic: {state['topic']}"),
        ]
    )

    print("ORCHESTRATOR: generated report sections")
    for index, section in enumerate(report_sections.sections, 1):
        print(f"{index}. {section.name}: {section.description}")

    return {"sections": report_sections.sections}


# ---------------------------------------------------------
# 4. Worker Node
#
# LangGraph runs this same node once per section by using Send.
# ---------------------------------------------------------
def write_section(state: WorkerState) -> dict:
    """Worker writes one section of the report."""
    section = state["section"]

    print(f"WORKER: writing section '{section.name}'")

    msg = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Write one report section using markdown formatting. "
                    "Follow the provided section name and description. "
                    "Include no preamble before the section."
                )
            ),
            HumanMessage(
                content=(
                    f"Section name: {section.name}\n"
                    f"Section description: {section.description}"
                )
            ),
        ]
    )

    return {"completed_sections": [msg.content]}


# ---------------------------------------------------------
# 5. Synthesizer Node
#
# Since workers already wrote full sections, the synthesizer joins
# them into one final markdown report.
# ---------------------------------------------------------
def synthesizer(state: State) -> dict:
    """Combine completed sections into the final report."""
    completed_report_sections = "\n\n---\n\n".join(state["completed_sections"])
    return {"final_report": completed_report_sections}


# ---------------------------------------------------------
# 6. Dynamic Worker Dispatch
#
# One Send object is created for each planned section.
# Each worker receives only the section it needs to write.
# ---------------------------------------------------------
def assign_workers(state: State) -> list[Send]:
    """Assign one worker to each section in the plan."""
    return [
        Send(
            "write_section",
            {"section": section},
        )
        for section in state["sections"]
    ]


# ---------------------------------------------------------
# 7. Build The Graph
# ---------------------------------------------------------
orchestrator_worker_builder = StateGraph(State)

orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("write_section", write_section)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["write_section"],
)
orchestrator_worker_builder.add_edge("write_section", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()


# ---------------------------------------------------------
# 8. Run It
# ---------------------------------------------------------
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "04_orchestrator_workers_report_sections_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(orchestrator_worker, graph_image_path)

    result = orchestrator_worker.invoke(
        {
            "topic": "Create a report on LLM scaling laws",
            "sections": [],
            "completed_sections": [],
            "final_report": "",
        }
    )

    print("\nFINAL REPORT\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
