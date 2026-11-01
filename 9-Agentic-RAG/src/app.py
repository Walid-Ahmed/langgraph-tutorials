# Streamlit chat UI for the knowledge-base chatbot.
#
# Each browser session gets one session_id, so with tracing on, every question in
# a conversation appears as one thread in LangSmith. Under each answer the page
# shows which tools the agent called, so you can watch it decide when to search.
#
# Prerequisites: .env with OPENAI_API_KEY (LANGSMITH_API_KEY and
# LANGSMITH_TRACING=true to trace), and the datasets/ folder; ingestion runs on first start.
# Run it with (from 9-Agentic-RAG):
#    streamlit run src/app.py
# Expected result: a chat page at http://localhost:8501. "Which sales rep covers Europe?"
# is answered from the knowledge base (tools: search_knowledge_base); "What's the
# weather in London?" uses get_weather and searches nothing.

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
        answer, tools = ask_with_tools(question, chat_history=st.session_state.messages,
                                       session_id=st.session_state.session_id)
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"tools: {', '.join(tools) or 'none'}")
    st.session_state.messages += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": answer}]
