# Run from the repository root:
#   python "6-Agents/05a_single_tool_date.py"
#
# What it does:
#   Gives the model one tool, get_today, and asks two questions. For "what day
#   is it?" the model requests the tool, ToolNode runs it, and a plain model
#   (no tools bound) writes the answer. For "what is the capital of Japan?" the
#   model answers at once and tools_condition ends the run.
#     START -> llm -> (tools_condition) -> tools -> answer -> END
#                                       \-> END
#
# What it demonstrates (Chapter 6 — Tools, §6.5, Listing 6.7):
#   - the docstring tells the model WHEN a tool applies; no if-statement in code
#   - ToolNode runs the requested call; tools_condition routes to "tools" or END
#   - one round of tools: there is no tools -> llm back edge, so not an agent yet
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (your date and the model's wording will differ):
#   Q: What day of the week is it today?
#     messages: human -> ai -> tool -> ai
#     A: Today is Wednesday, 30 September 2026.
#   Q: What is the capital of Japan?
#     messages: human -> ai
#     A: The capital of Japan is Tokyo.

from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


@tool
def get_today() -> str:
    """Return today's weekday and date, e.g. 'Wednesday, 30 September 2026'.

    Use it whenever the user asks what day or date it is today.
    """
    return date.today().strftime("%A, %d %B %Y")   # a str, as the signature promises


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_today])       # the model may now REQUEST get_today


def call_llm(state: MessagesState) -> dict:
    """Request the tool, or answer directly."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def answer(state: MessagesState) -> dict:
    """No tools bound here, so after one round the model can only answer."""
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode([get_today]))
builder.add_node("answer", answer)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)   # "tools" or END
builder.add_edge("tools", "answer")                     # forward only: one round
builder.add_edge("answer", END)
graph = builder.compile()

for question in ["What day of the week is it today?", "What is the capital of Japan?"]:
    result = graph.invoke({"messages": [HumanMessage(question)]})
    path = " -> ".join(m.type for m in result["messages"])
    print(f"Q: {question}")
    print(f"  messages: {path}")
    print(f"  A: {result['messages'][-1].content}")
