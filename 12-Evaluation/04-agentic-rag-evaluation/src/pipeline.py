# Ingestion: load the company files, chunk them, and store the chunks in Chroma.
#
# It runs once. If the collection already has chunks, it does nothing, so the
# chat UI and the evaluation can both call it safely. Delete .chroma/ after
# changing datasets/ to re-ingest.
#
# Prerequisites: OPENAI_API_KEY in .env (embeddings), and the datasets/ folder.
# Run it with (from 12-Evaluation/04-agentic-rag-evaluation):
#    python -m src.pipeline
# Expected result: "ingested N chunks from 11 files", or "ingestion skipped" on later runs.

from src import config
from src.chunking import fixed_size_chunk
from src.loaders import load_documents
from src.vectorstore import get_collection


def run_ingestion() -> None:
    collection = get_collection()
    if collection.count() > 0:
        print("ingestion skipped: the knowledge base already has chunks")
        return
    documents = load_documents(config.DATASETS_DIR)
    ids, texts, metadatas = [], [], []
    for doc in documents:
        for i, chunk in enumerate(fixed_size_chunk(doc.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            ids.append(f"{doc.source}::{i}")
            texts.append(chunk)
            metadatas.append({"source": doc.source, "chunk_index": i})  # the source is what the RAG metrics grade
    if not texts:
        raise SystemExit(f"no documents found in {config.DATASETS_DIR}")
    for start in range(0, len(texts), 100):  # Chroma embeds each batch in one API call
        collection.add(ids=ids[start:start + 100], documents=texts[start:start + 100],
                       metadatas=metadatas[start:start + 100])
    print(f"ingested {len(texts)} chunks from {len(documents)} files")


if __name__ == "__main__":
    run_ingestion()
