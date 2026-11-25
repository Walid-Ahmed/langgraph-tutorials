# Run from the repository root:
#   python "Exercise-Solutions/6-agents/ex1_word_count_tool.py"
#
# Exercise 6.2 — A second tool (Appendix B, Listing B.16)
#
# What it shows: the model chooses a tool by its description. We bind
# word_count next to Chapter 6's search_docs; asked how many words a phrase
# has, the model calls word_count because its docstring matches the request,
# not because of anything in the graph.
# Needs OPENAI_API_KEY in the repository-root .env file (the search tool
# embeds 5-Workflows/data once at start-up).
#
# Expected output (message list, abridged):
#   AIMessage tool_calls=[word_count(text='the quick brown fox')]
#   ToolMessage name=word_count content='4'
#   AIMessage content='The phrase has 4 words.'

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "5-Workflows" / "data"
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "6-Agents"))
from rag_index import build_vectorstore  # noqa: E402
from doc_tools import make_search_tool  # noqa: E402  (Chapter 6's search_docs)
from util import plot_graph  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_docs = make_search_tool(build_vectorstore(DOCS_DIR))


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text. Use this whenever the user asks how
    many words something contains; never estimate a count yourself."""
    return len(text.split())


llm_with_tools = llm.bind_tools([search_docs, word_count])


def call_llm(state: MessagesState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def main() -> None:
    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode([search_docs, word_count]))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")
    app = builder.compile()
    plot_graph(app, print_mermaid=False)

    result = app.invoke({"messages": [HumanMessage(
        "How many words are in the phrase 'the quick brown fox'?")]})
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            calls = ", ".join(f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})"
                              for c in msg.tool_calls)
            print(f"AIMessage tool_calls=[{calls}]")
        elif isinstance(msg, ToolMessage):
            print(f"ToolMessage name={msg.name} content={msg.content!r}")
        elif isinstance(msg, AIMessage):
            print(f"AIMessage content={msg.content!r}")


if __name__ == "__main__":
    main()
