# Run from 10-observability:
#   python document_workflow/main.py
#
# Demo for the five-node document pipeline (not create_agent).
# graph.py compiles the graph and loads document_workflow/.env (keys + project).
# This file: PNG, one invoke, print. LangSmith: env vars + run_name.

import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
OBS_FOLDER = PACKAGE.parent
REPO_ROOT = OBS_FOLDER.parent
sys.path.insert(0, str(OBS_FOLDER))
sys.path.append(str(REPO_ROOT))

from document_workflow.graph import graph
from util import plot_graph

QUESTION = "What is prompt injection?"
RUN_NAME = "Document workflow: prompt injection"


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Copy .env.example to "
            "document_workflow/.env and fill in keys."
        )

    graph_image_path = PACKAGE / "diagrams" / "document_workflow.png"
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(graph, str(graph_image_path), print_mermaid=False)

    print("\n" + "=" * 72)
    print(f"Q: {QUESTION}")

    # Custom state (not MessagesState). invoke returns the full ending state.
    state = {
        "question": QUESTION,
        "session_id": "demo",
        "steps_taken": [],
    }
    result = graph.invoke(
        state,
        config={"run_name": RUN_NAME, "recursion_limit": 25},
    )

    print(f"Nodes that ran: {result.get('steps_taken')}")
    print(f"Refined question: {result.get('refined_question')}")
    print("\nFINAL REPORT:")
    print((result.get("final_report") or "")[:800])

    project = os.getenv("LANGSMITH_PROJECT") or "document-workflow"
    print(f"\nOpen smith.langchain.com / project {project}")
    print(f"Trace title: {RUN_NAME}")


if __name__ == "__main__":
    main()
