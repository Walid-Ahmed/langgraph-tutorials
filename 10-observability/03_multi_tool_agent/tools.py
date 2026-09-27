# Toolbox for the research assistant.
#
# You register these functions. The model chooses which to call from the
# docstrings below (the triple-quoted text is sent in the tool schema).
# create_agent executes a requested tool; there is no ToolNode in this file.

import os

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool


def make_tools(vectorstore: FAISS):
    """Bind both tools to this run's FAISS index (closure)."""

    @tool
    def search_local_docs(query: str) -> str:
        """
        Search local employee text files (name, salary, address, role).

        Use when the question is about people or HR facts in those files.

        Call at most once per question, then answer the user.
        """
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant documents found."
        response = "\n\n".join(
            f"[Chunk {i + 1}]\n{doc.page_content[:700]}"
            for i, doc in enumerate(docs)
        )
        return response[:2500]

    @tool
    def google_search(query: str) -> str:
        """
        Search the live web for recent information.

        Use for current events, news, regulations, and recent AI developments.

        Call at most once per question, then answer the user.
        """
        if not os.getenv("SERPER_API_KEY"):
            return (
                "Web search unavailable: SERPER_API_KEY is missing. "
                "Answer from general knowledge or say you cannot fetch live news."
            )
        try:
            result = GoogleSerperAPIWrapper().run(query)
            if not result:
                return "No search results found."
            return str(result)[:2500]
        except Exception as exc:
            return f"Search failed: {exc}"

    return [search_local_docs, google_search]
