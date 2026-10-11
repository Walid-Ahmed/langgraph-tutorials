# Research Assistant tools - the two tools used by graph.py in this folder.
#
# This module defines search_local_docs (keyword search over the local LLM
# production guide) and web_search (a live Google search through Serper). It is
# imported by graph.py and is not meant to be run on its own.
#
# Why this matters:
# - Both functions are LangChain @tool objects, and a @tool is a Runnable. When
#   a node calls one with .invoke(), LangSmith records it as a tool run nested
#   under that node, with no tracing code here.
# - There is deliberately NO @traceable and no LangSmith import in this file.
#
# Before you use this:
# 1. Install this folder's dependencies (from the 10-observability folder):
#    pip install -r requirements.txt
# 2. SERPER_API_KEY must already be in the environment when this module is
#    imported, because GoogleSerperAPIWrapper() reads it on construction.
#    run_traced.py calls load_dotenv() before importing graph.py for this reason.
# 3. DOC_PATH is relative to the 10-observability folder, so run from there.

import re
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool

DOC_PATH = "data/llm_production_guide.txt"
serper = GoogleSerperAPIWrapper()                        # reads SERPER_API_KEY


@tool
def search_local_docs(query: str, top_k: int = 3) -> str:
    """Search the local LLM production guide by keyword overlap and return the best paragraphs."""
    text = open(DOC_PATH, encoding="utf-8").read()
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    keywords = set(re.findall(r"\b\w{4,}\b", query.lower()))

    ranked = sorted(paragraphs, key=lambda p: sum(kw in p.lower() for kw in keywords), reverse=True)
    return "\n\n".join(ranked[:top_k]) if ranked else "No relevant local sections found."


@tool
def web_search(query: str) -> str:
    """Search the live web for recent information."""
    result = serper.run(query)
    return str(result)[:2500] if result else "No search results found."
