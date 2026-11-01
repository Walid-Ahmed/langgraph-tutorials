# The persistent Chroma collection the knowledge-base tool searches.
#
# Chroma embeds documents and queries with the same OpenAI embedding model, so
# a question and the chunk that answers it land close together.
# This module is imported, not run.

from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from src import config


@lru_cache(maxsize=1)
def get_collection():
    """Open (or create) the collection once per process."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    embed = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY, model_name=config.EMBEDDING_MODEL,
    )
    return client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME, embedding_function=embed)
