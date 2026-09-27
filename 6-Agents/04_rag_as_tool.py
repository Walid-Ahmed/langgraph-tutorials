# Run from the repository root:
#   python "6-Agents/04_rag_as_tool.py"
#
# What it does:
#   Builds the same local FAISS index as 5-Workflows/01_rag_retrieve_generate.py,
#   but instead of wiring retrieval into every run, wraps it as a TOOL. The
#   model decides whether to search, how many times, and with what query.
#
# What it demonstrates (Chapter 8 — Agents; the tool itself is taught in Chapter 7):
#   - a tool with a Pydantic args schema: the model sees the schema, and bad
#     arguments are rejected before your code runs
#   - a tool built by a factory so it closes over a prebuilt index
#   - content_and_artifact: text for the model, structured sources for your code
#   - the prebuilt ToolNode + tools_condition edge
#   - parallel tool calls (two searches in one model turn)
#   - errors as information (bad args go back to the model) vs. transient
#     failures (retried by a RetryPolicy on the tools node)
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env (chat model + embeddings).
#   Docs folder: 5-Workflows/data/ (llm_production_guide.txt).
#
# Expected output (wording varies run to run; the tool behaviour should not):
#   Q1 (about the guide)      -> 1 call to search_docs, answer citing chunks
#   Q2 (two topics at once)   -> 2 search_docs calls in the SAME turn (parallel)
#   Q3 (general knowledge)    -> 0 tool calls, direct answer
#   Q4 (simulated outage)     -> first search attempt fails, RetryPolicy retries,
#                                answer still arrives

import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "5-Workflows" / "data"
sys.path.append(str(REPO_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))
from rag_index import build_vectorstore  # noqa: E402
from util import plot_graph  # noqa: E402

load_dotenv()


from doc_tools import SIMULATE_OUTAGE, make_search_tool  # noqa: E402  (shared with 05_tools_single_round.py)


# ---------------------------------------------------------
# 2. State — Chapter 4's message list, nothing more
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = (
    "You answer questions for an engineering team. "
    "For anything covered by the internal LLM production guide, call search_docs "
    "and answer ONLY from the passages it returns, citing them as [Chunk N]. "
    "If a question has several distinct parts, search for each part separately. "
    "For general knowledge, answer directly without searching."
)


# ---------------------------------------------------------
# 3. Graph — the agent loop, built from prebuilt parts
# ---------------------------------------------------------
def build_agent(vectorstore, llm=None):
    search_docs = make_search_tool(vectorstore)
    tools = [search_docs]

    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)  # advertise; the graph executes

    def call_llm(state: AgentState) -> dict:
        response = llm_with_tools.invoke(
            [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        )
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("llm", call_llm)
    # ToolNode's default error handler returns bad-argument errors to the model
    # and re-raises everything else, so the RetryPolicy below sees transient
    # failures (ConnectionError) and re-runs the node.
    builder.add_node(
        "tools",
        ToolNode(tools),
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5,
                                 retry_on=ConnectionError),
    )
    builder.add_edge(START, "llm")
    # tools_condition: "tools" if the last AIMessage has tool_calls, else END.
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")  # the back edge that makes it an agent
    return builder.compile()


# ---------------------------------------------------------
# 4. A small trace printer: what did the model decide?
# ---------------------------------------------------------
def summarize_run(messages: list) -> None:
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            calls = ", ".join(
                f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})"
                for c in msg.tool_calls
            )
            print(f"  model turn -> {len(msg.tool_calls)} tool call(s): {calls}")
        elif isinstance(msg, ToolMessage):
            cited = [s["chunk"] for s in (msg.artifact or [])]
            print(f"  tool result -> {msg.name}: {len(cited)} passage(s), status={msg.status}")
    tool_turns = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    if tool_turns == 0:
        print("  model turn -> answered directly, no tools")
    print(f"\n  Answer: {messages[-1].content}\n")


def main() -> None:
    print(f"Building FAISS index from {DOCS_DIR} ...")
    vectorstore = build_vectorstore(DOCS_DIR)  # once, before any question
    agent = build_agent(vectorstore)

    graph_image_path = Path(__file__).resolve().parent / "diagrams" / "04_rag_as_tool.png"
    graph_image_path.parent.mkdir(exist_ok=True)
    plot_graph(agent, graph_image_path)

    questions = [
        ("Q1", "What is prompt injection and how do we defend against it?", 0),
        ("Q2", "Compare LLM01 and LLM02 from the OWASP list.", 0),
        ("Q3", "What is the capital of Canada?", 0),
        ("Q4", "What does the gateway layer do?", 1),  # 1 simulated failure
    ]
    for label, question, failures in questions:
        SIMULATE_OUTAGE["remaining_failures"] = failures
        print("=" * 70)
        print(f"{label}: {question}")
        print("=" * 70)
        result = agent.invoke(
            {"messages": [HumanMessage(question)]},
            config={"recursion_limit": 10},  # a hard cap on laps
        )
        summarize_run(result["messages"])


if __name__ == "__main__":
    main()
