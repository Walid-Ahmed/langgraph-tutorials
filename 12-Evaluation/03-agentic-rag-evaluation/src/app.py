# Streamlit chat UI for the knowledge-base chatbot: the "production" stage.
#
# Each browser session gets one session_id, so every question in a conversation
# appears as one thread in LangSmith, and each answer is scored online in the
# background (toxicity, bias, answer relevancy).
#
# Prerequisites: .env with OPENAI_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true,
# and the datasets/ folder (ingestion runs on first start).
# Run it with (from 12-Evaluation/03-agentic-rag-evaluation):
#    streamlit run src/app.py
# Expected result: a chat page at http://localhost:8501; each question appears as a
# trace in your LangSmith project with toxicity, bias_fairness and answer_relevancy feedback.

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "src" imports work under streamlit

from src.agents import ask_with_tools
from src.pipeline import run_ingestion

st.set_page_config(page_title="Company knowledge base", page_icon="💬")
st.title("Company knowledge base")

if "session_id" not in st.session_state:
    run_ingestion()
    st.session_state.session_id = f"chatbot-{uuid.uuid4()}"
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if question := st.chat_input("Ask a question..."):
    st.chat_message("user").write(question)
    with st.spinner("Thinking..."):
        reply = ask_with_tools(question, chat_history=st.session_state.messages,
                               session_id=st.session_state.session_id)
    st.chat_message("assistant").write(reply.answer)
    st.session_state.messages += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": reply.answer}]
