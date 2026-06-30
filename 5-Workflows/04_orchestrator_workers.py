import sys
from operator import add
from pathlib import Path
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()


# ---------------------------------------------------------
# 1. State Definitions
#
# OverallState is the shared graph state.
# WorkerState is the smaller state sent to each worker.
#
# worker_findings uses a reducer: Annotated[List[dict], add]
# This matters because many workers return findings in parallel.
# The reducer tells LangGraph to append/merge those lists instead
# of overwriting one worker's result with another worker's result.
# ---------------------------------------------------------
class OverallState(TypedDict):
    research_topic: str
    sources: List[str]
    worker_findings: Annotated[List[dict], add]
    final_report: str


class WorkerState(TypedDict):
    source: str
    worker_id: int
    research_topic: str


class ResearchPlan(BaseModel):
    """Structured plan created by the orchestrator."""

    sources: List[str] = Field(
        description="List of specific research sources/aspects to investigate",
        max_length=5,
    )
    reasoning: str = Field(
        description="Brief explanation of why these sources were chosen",
    )


llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------
# 2. Orchestrator Node
#
# The orchestrator does not do all the research itself.
# It creates a plan: which source/aspect should each worker study?
# ---------------------------------------------------------
def plan_research(state: OverallState) -> dict:
    """Plan the research strategy and choose worker focus areas."""
    print("\n" + "=" * 70)
    print("ORCHESTRATOR: planning research strategy")
    print("=" * 70)
    print(f"Topic: {state['research_topic']}\n")

    planner_llm = llm.with_structured_output(ResearchPlan)

    prompt = f"""
    You are a research strategist planning a comprehensive investigation.

    Research Topic: {state['research_topic']}

    Generate between 3-5 specific research sources or aspects to investigate.
    Do not generate more than 5 sources.

    Each source should be:
    - specific and focused on a distinct aspect
    - relevant to the overall topic
    - complementary to other sources with minimal overlap
    - concrete enough to guide targeted research

    Examples of good sources:
    - Clinical trial results and efficacy data
    - Economic impact and cost-benefit analysis
    - Regulatory framework and compliance requirements
    - Patient outcomes and quality of life metrics
    - Industry adoption rates and market trends

    Generate sources that provide comprehensive coverage of the topic.
    """

    research_plan = planner_llm.invoke(prompt)

    print(f"Orchestrator generated {len(research_plan.sources)} research sources")
    for index, source in enumerate(research_plan.sources, 1):
        print(f"{index}. {source}")

    print(f"\nReasoning: {research_plan.reasoning}")
    print("Preparing to dispatch workers...\n")

    return {"sources": research_plan.sources}


# ---------------------------------------------------------
# 3. Worker Node
#
# LangGraph can run this same node multiple times with different
# WorkerState values. Each worker receives one source/aspect.
# ---------------------------------------------------------
def research_worker(state: WorkerState) -> dict:
    """Research one source/aspect selected by the orchestrator."""
    worker_id = state["worker_id"]
    source = state["source"]

    print(f"WORKER {worker_id}: researching '{source}'...")

    prompt = f"""
    You are a specialized researcher investigating: {state['research_topic']}

    Your specific focus area: {source}

    Conduct focused research on this aspect and provide:

    1. KEY FINDINGS
    - 3-5 important discoveries or facts

    2. DATA & STATISTICS
    - relevant numbers, percentages, or quantitative information

    3. INSIGHTS & ANALYSIS
    - what this information means
    - how it relates to the broader topic

    4. SOURCES & CREDIBILITY
    - types of sources you would consult

    5. IMPLICATIONS
    - why this matters for understanding the overall topic

    Be specific, factual, and focused on this particular aspect.
    """

    response = llm.invoke(prompt).content

    findings = {
        "worker_id": worker_id,
        "source": source,
        "content": response,
    }

    print(f"WORKER {worker_id}: research complete\n")

    return {"worker_findings": [findings]}


# ---------------------------------------------------------
# 4. Synthesizer Node
#
# The synthesizer runs after the workers finish.
# It reads all worker_findings and creates one unified report.
# ---------------------------------------------------------
def synthesize_report(state: OverallState) -> dict:
    """Combine all worker findings into one final report."""
    print("=" * 70)
    print("SYNTHESIZER: combining insights from all workers")
    print("=" * 70)

    print(f"Processing findings from {len(state['worker_findings'])} workers")

    all_findings = "\n\n" + "=" * 70 + "\n\n"
    all_findings += "\n\n".join(
        [
            f"RESEARCH AREA {finding['worker_id']}: {finding['source']}\n"
            f"{'-' * 70}\n{finding['content']}"
            for finding in state["worker_findings"]
        ]
    )

    prompt = f"""
    You are synthesizing a comprehensive research report on: {state['research_topic']}

    You received detailed findings from {len(state['worker_findings'])} specialized researchers.

    RESEARCH FINDINGS:
    {all_findings}

    Create a cohesive, well-structured research report of 500-700 words with:

    1. EXECUTIVE SUMMARY
    2. INTRODUCTION
    3. KEY FINDINGS
    4. ANALYSIS & SYNTHESIS
    5. IMPLICATIONS
    6. CONCLUSIONS

    Important:
    - write this as one unified report
    - do not simply paste each worker section separately
    - integrate findings naturally across themes
    - use specific examples and data from the findings
    - make it professional and authoritative
    """

    final_report = llm.invoke(prompt).content

    print("SYNTHESIZER: final report complete\n")

    return {"final_report": final_report}


# ---------------------------------------------------------
# 5. Dynamic Worker Dispatch
#
# This is the key orchestrator-workers idea.
# The orchestrator creates a list of sources, then this function
# returns one Send(...) object per source.
#
# Each Send tells LangGraph:
# "Run research_worker with this worker-specific state."
# ---------------------------------------------------------
def create_research_workers(state: OverallState) -> list[Send]:
    """Create one worker call for each planned research source."""
    print("DISPATCHER: creating research workers dynamically...")

    return [
        Send(
            "research_worker",
            {
                "source": source,
                "worker_id": index + 1,
                "research_topic": state["research_topic"],
            },
        )
        for index, source in enumerate(state["sources"])
    ]


# ---------------------------------------------------------
# 6. Build The Graph
# ---------------------------------------------------------
builder = StateGraph(OverallState)

builder.add_node("orchestrator", plan_research)
builder.add_node("research_worker", research_worker)
builder.add_node("synthesizer", synthesize_report)

builder.add_edge(START, "orchestrator")

builder.add_conditional_edges(
    "orchestrator",
    create_research_workers,
    ["research_worker"],
)

builder.add_edge("research_worker", "synthesizer")
builder.add_edge("synthesizer", END)

graph = builder.compile()


# ---------------------------------------------------------
# 7. Run It
# ---------------------------------------------------------
def main() -> None:
    graph_image_path = (
        Path(__file__).resolve().parent
        / "diagrams"
        / "04_orchestrator_workers_graph.png"
    )
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(graph, graph_image_path)

    topic = "Renewable energy adoption barriers in developing countries"

    print("=" * 70)
    print(f"Topic: {topic}")
    print("=" * 70)

    result = graph.invoke(
        {
            "research_topic": topic,
            "sources": [],
            "worker_findings": [],
            "final_report": "",
        }
    )

    print("=" * 70)
    print("FINAL SYNTHESIZED RESULT")
    print("=" * 70)
    print(result["final_report"])


if __name__ == "__main__":
    main()
