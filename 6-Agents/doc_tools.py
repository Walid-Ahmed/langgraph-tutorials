# Shared tool for tutorial 6 (and book Chapters 7-8).
#
# search_docs searches the local FAISS index built by repo-root rag_index.py.
# It lives in its own module so that the single-round tools example
# (05_tools_single_round.py, Chapter 7) and the agent example
# (04_rag_as_tool.py, Chapter 8) use exactly the same tool.
#
# Nothing to run here directly; import make_search_tool from the examples.

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. The tool's contract
#
# The model never sees the function body. It sees the tool name, the
# docstring, and this schema. Field descriptions tell it WHAT to pass;
# ge/le tell it the legal range, and Pydantic enforces that range before
# the search runs.
# ---------------------------------------------------------
class SearchDocsInput(BaseModel):
    query: str = Field(
        description="A focused search phrase, e.g. 'prompt injection defenses'. "
        "Rewrite the user's question into keywords; do not paste it verbatim."
    )
    k: int = Field(
        default=3, ge=1, le=5,
        description="How many passages to return (1-5). Use more for broad questions.",
    )


# Flip to True (or let main() do it for Q4) to simulate a flaky vector store.
SIMULATE_OUTAGE = {"remaining_failures": 0}


def make_search_tool(vectorstore):
    """Build a search tool that closes over an already-built index.

    Same idea as Chapter 6: build the expensive index once, outside the graph,
    and let the node (here, the tool) only read it.
    """

    @tool(args_schema=SearchDocsInput, response_format="content_and_artifact")
    def search_docs(query: str, k: int = 3):
        """Search the internal 'LLM Production Security and Observability Guide'.

        Use this for questions about LLM security threats (OWASP LLM Top 10,
        prompt injection), guardrails, gateways, evaluation, observability,
        or production deployment. Do NOT use it for general knowledge,
        arithmetic, or current events.
        """
        if SIMULATE_OUTAGE["remaining_failures"] > 0:
            SIMULATE_OUTAGE["remaining_failures"] -= 1
            # A transient failure: raised, not returned, so RetryPolicy can retry it.
            raise ConnectionError("vector store temporarily unavailable")

        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            # "Nothing found" is an answer, not an exception: tell the model.
            return "No relevant passages found in the guide.", []

        text = "\n\n".join(
            f"[Chunk {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )
        # The artifact never reaches the model; it is for your code (logging,
        # citations, UI) and survives as ToolMessage.artifact.
        sources = [
            {
                "chunk": i + 1,
                "source": Path(doc.metadata.get("source", "?")).name,
                "preview": doc.page_content[:60].replace("\n", " "),
            }
            for i, doc in enumerate(docs)
        ]
        return text, sources

    return search_docs
