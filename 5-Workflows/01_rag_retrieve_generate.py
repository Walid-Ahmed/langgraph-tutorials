# Run from the repository root:
#   python "5-Workflows/01_rag_retrieve_generate.py"
#
# Local RAG as a workflow (not an agent): you wire retrieve then generate.
# Every question takes the same path. Tutorial 6 / 10 can wrap the same
# index as a tool so the *model* decides whether to retrieve.

import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

FOLDER = Path(__file__).resolve().parent
DOCS_DIR = FOLDER / "data"
sys.path.append(str(FOLDER.parent))
from util import plot_graph
from rag_index import build_vectorstore

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------
# 1. State
#
# question  : user input
# retrieved : chunks written by the retrieve node
# answer    : final text written by the generate node
# ---------------------------------------------------------
class RAGState(TypedDict):
    question: str
    retrieved: str
    answer: str


# ---------------------------------------------------------
# 2. Graph — retrieve always runs, then generate always runs
# ---------------------------------------------------------
def build_graph(vectorstore):
    def retrieve(state: RAGState) -> dict:
        docs = vectorstore.similarity_search(state["question"], k=3)
        if not docs:
            return {"retrieved": "No relevant documents found."}
        text = "\n\n".join(
            f"[Chunk {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )
        print("=== STEP 1: Retrieved chunks ===")
        print(text[:400] + "...\n")
        return {"retrieved": text}

    def generate(state: RAGState) -> dict:
        prompt = f"""
        Answer the question using only the retrieved context.
        If the context is not enough, say so.

        Question: {state["question"]}

        Context:
        {state["retrieved"]}
        """
        answer = llm.invoke(prompt).content
        print("=== STEP 2: Answer ===")
        print(answer[:200] + "...\n")
        return {"answer": answer}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to the repository-root .env file."
        )

    print(f"Building FAISS index from {DOCS_DIR} ...")
    vectorstore = build_vectorstore(DOCS_DIR)
    graph = build_graph(vectorstore)

    graph_image_path = FOLDER / "diagrams" / "01_rag_retrieve_generate_graph.png"
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(graph, graph_image_path)

    result = graph.invoke(
        {
            "question": (
                "What is prompt injection and how do we defend against it?"
            ),
            "retrieved": "",
            "answer": "",
        }
    )

    print("\n" + "=" * 50)
    print("FINAL ANSWER")
    print("=" * 50)
    print(result["answer"])


if __name__ == "__main__":
    main()
