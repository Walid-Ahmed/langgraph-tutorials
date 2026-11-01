# The agentic RAG chatbot: a ReAct agent that decides when to search the knowledge base.
#
# Why this matters:
# - create_agent builds a LangGraph graph (a model node and a tool node in a loop),
#   so this is the Chapter 7 agent loop with a vector store behind one tool.
# - Retrieval is a tool the model chooses to call. A company question triggers
#   search_knowledge_base; a weather question calls get_weather and searches nothing.
#   That choice is what makes this agentic RAG rather than classic RAG.
# - With LANGSMITH_TRACING=true every question is traced (Chapter 13), and the
#   session_id groups a conversation's questions into one LangSmith thread.
#
# Prerequisites: OPENAI_API_KEY in .env and an ingested knowledge base
# (python -m src.pipeline). Imported by app.py, not run.

from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from src import config
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a helpful assistant for a company's internal chatbot.
You have access to tools:
- search_knowledge_base: use it to answer questions about company data
  (CRM accounts, opportunities, sales reps, projects, revenue, proposals).
- get_weather: use it for weather questions.

Only call a tool when it is relevant to the question. Answer general questions
directly. Ground every answer about company data in the knowledge base results,
and say so when you could not find the information.
Records refer to people, accounts and projects by ID (REP01, ACC004, PRJ004).
When the answer needs a name, search again for that ID and answer with the name.
When the question asks for every record that matches a condition (won, delayed,
over budget), check that condition on each row before you include it."""


# Build the agent once: a model node and a tool node in a loop (create_agent).
@lru_cache(maxsize=1)
def get_agent():
    model = ChatOpenAI(model=config.AGENT_MODEL, temperature=0)
    return create_agent(model, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


# Ask the agent one question; return the answer and the tools it called, in order.
def ask_with_tools(question: str, chat_history: list[dict] | None = None,
                   session_id: str | None = None) -> tuple[str, list[str]]:
    messages = [*(chat_history or []), {"role": "user", "content": question}]
    result = get_agent().invoke({"messages": messages}, config={
        "run_name": "chatbot",
        "tags": ["chatbot"],
        "metadata": {"session_id": session_id} if session_id else {},
    })
    new_messages = result["messages"][len(messages):]   # only this turn, not the history
    tools = [m.name or "" for m in new_messages if isinstance(m, ToolMessage)]
    return result["messages"][-1].text, tools
