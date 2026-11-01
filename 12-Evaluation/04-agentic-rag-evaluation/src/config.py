# Settings shared by the chatbot (src/) and the evaluation harness (evals/).
#
# Keeping paths, model names and chunk sizes in one place means the offline
# evaluation always measures the same configuration the chat UI runs.
#
# Prerequisites: a .env file in this folder (or a parent folder) with
# OPENAI_API_KEY, LANGSMITH_API_KEY and LANGSMITH_TRACING=true.
# This module is imported, not run.

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR.parents[1] / "9-Agentic-RAG" / "datasets"  # shared with the Chapter 12 app
CHROMA_PERSIST_DIR = ROOT_DIR / ".chroma"      # the vector store, built on first run
CHROMA_COLLECTION_NAME = "chatbot_knowledge_base"

CHUNK_SIZE = 1000                              # characters per chunk
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.2)          # 20% overlap so a fact split at a boundary survives

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o-mini"                    # the chatbot
JUDGE_MODEL = "gpt-4o"                         # the LLM-as-judge: at least as strong as the system it grades
EMBEDDING_MODEL = "text-embedding-3-small"
