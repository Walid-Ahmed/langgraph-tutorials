# Compiled create_agent graph. Import this from a server; do not copy main.py.
#
# SYSTEM_PROMPT is role + a call budget. It does not list tools. The model
# picks tools from the @tool docstrings in tools.py.
#
# create_agent returns a compiled LangGraph whose state is MessagesState
# ({"messages": [...]}).

from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from tools import make_tools

SYSTEM_PROMPT = """You are a research assistant.

Call a tool only if needed. At most two tool calls. Never call the same
tool twice. After tool results, answer. Do not keep searching.
"""


def build_research_assistant(
    vectorstore: FAISS,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
):
    """Return a compiled LangGraph. Caller owns indexing and env."""
    return create_agent(
        model=ChatOpenAI(model=model, temperature=temperature),
        tools=make_tools(vectorstore),
        system_prompt=SYSTEM_PROMPT,
    )
