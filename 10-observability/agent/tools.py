"""Utility tools used by agent nodes."""
import os
import re
from typing import List


_DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "llm_production_guide.txt")


def load_document() -> str:
    """Load the full knowledge-base text file."""
    with open(_DOC_PATH, encoding="utf-8") as f:
        return f.read()


def search_document(query: str, top_k: int = 3) -> List[str]:
    """
    Simple keyword search over the text document.
    Splits into paragraphs, scores by keyword overlap, returns top_k.
    No vector DB needed — keeps the demo focused on LangSmith tracing.
    """
    doc = load_document()
    paragraphs = [p.strip() for p in doc.split("\n\n") if len(p.strip()) > 80]

    keywords = set(re.findall(r"\b\w{4,}\b", query.lower()))

    def score(para: str) -> int:
        text_lower = para.lower()
        return sum(1 for kw in keywords if kw in text_lower)

    ranked = sorted(paragraphs, key=score, reverse=True)
    return ranked[:top_k]
