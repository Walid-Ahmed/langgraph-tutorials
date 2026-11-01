# Turn the files under datasets/ into plain text the chunker can split.
#
# CSV rows become "column: value" lines, so a chunk keeps the column names next
# to the values and a question like "who covers Europe?" can match the row.
# This module is imported by pipeline.py, not run.

import csv
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


# One file's text, and the path it came from.
@dataclass
class LoadedDocument:
    text: str
    source: str   # path relative to datasets/, e.g. "crm_exports/accounts.csv"


# A CSV file as text: one line per row, written as "column: value" pairs.
def _load_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        return "\n".join(", ".join(f"{key}: {value}" for key, value in row.items())
                         for row in csv.DictReader(f))


# A PDF's text, page by page; a page with no text becomes an empty line.
def _load_pdf(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


# Any other file, read as plain text.
def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_LOADERS = {".csv": _load_csv, ".pdf": _load_pdf, ".txt": _load_text, ".md": _load_text}


def load_documents(datasets_dir: Path) -> list[LoadedDocument]:
    """Load every supported file under datasets_dir, in a stable order."""
    documents = []
    for path in sorted(datasets_dir.rglob("*")):
        loader = _LOADERS.get(path.suffix.lower())
        if path.is_file() and loader:
            text = loader(path)
            if text.strip():
                documents.append(LoadedDocument(text, path.relative_to(datasets_dir).as_posix()))
    return documents
