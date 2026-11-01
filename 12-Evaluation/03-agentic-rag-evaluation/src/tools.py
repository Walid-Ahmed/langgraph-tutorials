# The agent's two tools: a knowledge-base search and a dummy weather lookup.
#
# The search result tags every chunk with "[source: <file>]". The model reads it
# to ground its answer, and the evaluation reads it to know which files the
# agent actually retrieved. This module is imported, not run.

from langchain_core.tools import tool

from src.vectorstore import get_collection

SOURCE_TAG = "[source: "


@tool
def search_knowledge_base(query: str, n_results: int = 4) -> str:
    """Search the internal knowledge base (CRM exports, projects, revenue, proposal
    documents). Use it for any question about company data: accounts, opportunities,
    sales reps, projects, revenue or proposals."""
    collection = get_collection()
    if collection.count() == 0:
        return "The knowledge base is empty."
    results = collection.query(query_texts=[query], n_results=n_results)
    documents, metadatas = results["documents"][0], results["metadatas"][0]
    if not documents:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(f"{SOURCE_TAG}{meta['source']}]\n{text}"
                              for text, meta in zip(documents, metadatas))


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location (dummy data, for the demo)."""
    return f"The weather in {location} is sunny with a high of 24°C (dummy data)."


ALL_TOOLS = [search_knowledge_base, get_weather]
