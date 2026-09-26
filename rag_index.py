# Shared local-RAG helper (tutorials 5 and 10).
#
# Sibling of util.py: plot_graph stays visualization-only so tutorials 1–4
# do not import FAISS. This module loads every .txt file in a folder, splits,
# embeds, and returns a FAISS index. Build the index once; graph nodes only
# call similarity_search.

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_vectorstore(docs_dir: str | Path) -> FAISS:
    """Load, split, and embed every .txt file in docs_dir into a FAISS index."""
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise SystemExit(f"Missing docs folder: {docs_dir}")

    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    raw_docs = loader.load()
    if not raw_docs:
        raise SystemExit(f"No .txt files found in {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)
