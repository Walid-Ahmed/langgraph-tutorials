# Settings for the knowledge-base chatbot: paths, model names and chunk sizes.
#
# Prerequisites: a .env file in this folder (or a parent folder) with
# OPENAI_API_KEY, and LANGSMITH_API_KEY plus LANGSMITH_TRACING=true to trace runs.
# This module is imported, not run.

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"           # the company files the chatbot answers from
CHROMA_PERSIST_DIR = ROOT_DIR / ".chroma"      # the vector store, built on first run
CHROMA_COLLECTION_NAME = "chatbot_knowledge_base"

CHUNK_SIZE = 1000                              # characters per chunk
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.2)          # 20% overlap so a fact split at a boundary survives

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
