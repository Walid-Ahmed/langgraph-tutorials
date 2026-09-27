# Run from 10-observability:
#   python 03_multi_tool_agent/main.py
#
# Demo harness for the research assistant. The compiled graph lives in
# research_assistant.py. This file: env, FAISS, one PNG, three invokes, print.
#
# Tracing is the same as 01/02: LANGSMITH_* in this folder's .env (complete
# file: keys + LANGSMITH_PROJECT). There is no langsmith import and no
# @traceable. Load .env before importing LangChain so the SDK can patch.

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PACKAGE = Path(__file__).resolve().parent
OBS_FOLDER = PACKAGE.parent
REPO_ROOT = OBS_FOLDER.parent
sys.path.append(str(REPO_ROOT))

load_dotenv(PACKAGE / ".env")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState

from rag_index import build_vectorstore
from util import plot_graph

from research_assistant import build_research_assistant

# Each tuple is (user question, LangSmith trace title, expected tools).
# The hint is printed only; it is not passed to the graph.
QUERIES = [
    (
        "What is Bob Martinez’s salary and department?",
        "Multi-Tool: local docs",
        "search_local_docs (expected, not enforced)",
    ),
    (
        "What are the latest AI regulations passed in 2025?",
        "Multi-Tool: web search",
        "google_search (expected, not enforced)",
    ),
    (
        "What is Charlie Kim's role and salary, and how does that compare to typical senior data engineer pay in 2025?",
        "Multi-Tool: both",
        "both tools (expected, not enforced)",
    ),
]


def tools_used(messages: list) -> list[str]:
    """Terminal helper only. LangSmith already shows tool runs."""
    names = [
        msg.name for msg in messages if isinstance(msg, ToolMessage) and msg.name
    ]
    return list(dict.fromkeys(names))


def final_answer(messages: list) -> str:
    """Terminal helper: last AI text from state. invoke() itself returns state."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Copy .env.example to "
            "03_multi_tool_agent/.env and fill in keys."
        )

    docs_dir = PACKAGE / "local_RAG"
    print(f"Building FAISS index from {docs_dir} ...")
    vectorstore = build_vectorstore(docs_dir)
    agent = build_research_assistant(vectorstore)

    graph_image_path = PACKAGE / "diagrams" / "03_multi_tool_agent.png"
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(agent, str(graph_image_path), print_mermaid=False)

    for question, run_name, hint in QUERIES:
        print("\n" + "=" * 72)
        print(f"Q: {question}")
        print(f"Hint (not used by the graph): {hint}")

        # create_agent state is MessagesState. The required key is "messages"
        # (unlike file 01, where we named team/answer ourselves).
        state: MessagesState = {
            "messages": [HumanMessage(content=question)]
        }
        # invoke returns the ending state dict, not a plain answer string.
        # run_name is the LangSmith *trace title*. tracing itself is env vars.
        result = agent.invoke(
            state,
            config={
                "run_name": run_name,
                "recursion_limit": 10,
            },
        )

        used = tools_used(result["messages"])
        answer = final_answer(result["messages"])
        print(f"Tools the agent actually called: {used or '(none)'}")
        print("\nANSWER:")
        print(answer[:800])

    project = os.getenv("LANGSMITH_PROJECT") or "10-observability"
    print(f"\nOpen smith.langchain.com / project {project}")
    print("and compare the three traces named Multi-Tool: ...")


if __name__ == "__main__":
    main()
