# Run from the repository root:
#   python "6-Agents/05_tools_single_round.py"
#
# What it does:
#   Takes the search_docs tool (doc_tools.py) through every step of a tool
#   call, one step at a time, WITHOUT an agent loop:
#     1. print the contract the model sees (name, description, JSON schema)
#     2. bind the tool and let the model PROPOSE a call (tool_calls)
#     3. execute that call by hand and get a ToolMessage back
#     4. run a small graph with exactly one round of tools:
#        START -> llm -> (tools_condition) -> tools -> answer -> END
#
# What it demonstrates (Chapter 7 — Tools and Tool Calling):
#   - a tool is a contract: the model sees only name + description + schema
#   - bind_tools advertises; the model proposes; your code executes
#   - tool_call_id pairs each ToolMessage with its request
#   - content_and_artifact: text for the model, structured data for your code
#   - ToolNode + tools_condition in a graph with NO back edge (not an agent yet)
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env (chat model + embeddings).
#
# Expected output (model wording varies; the structure should not):
#   Step 1 -> the JSON contract for search_docs (deterministic)
#   Step 2 -> one tool call, e.g. search_docs(query='prompt injection defenses')
#   Step 3 -> a ToolMessage with the same tool_call_id, 3 passages, 3 sources
#   Step 4 -> Q1 goes llm -> tools -> answer; Q2 (general knowledge) goes llm -> END

import json
import sys
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "5-Workflows" / "data"
sys.path.append(str(REPO_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))
from rag_index import build_vectorstore  # noqa: E402
from doc_tools import make_search_tool  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

SYSTEM_PROMPT = (
    "Answer questions for an engineering team. For anything covered by the "
    "internal LLM production guide, call search_docs. For general knowledge, "
    "answer directly."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_single_round_graph(search_docs, llm):
    """One round of tools, then a final answer. There is NO tools -> llm edge."""
    llm_with_tools = llm.bind_tools([search_docs])

    def call_llm(state: State) -> dict:
        return {"messages": [llm_with_tools.invoke(
            [SystemMessage(SYSTEM_PROMPT)] + state["messages"])]}

    def answer(state: State) -> dict:
        # Plain llm (no tools bound): after one round it can only answer.
        return {"messages": [llm.invoke(
            [SystemMessage("Answer only from the tool results above, citing [Chunk N].")]
            + state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode([search_docs]))
    builder.add_node("answer", answer)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)   # "tools" or END
    builder.add_edge("tools", "answer")                     # forward only: no loop
    builder.add_edge("answer", END)
    return builder.compile()


def main() -> None:
    print(f"Building FAISS index from {DOCS_DIR} ...")
    search_docs = make_search_tool(build_vectorstore(DOCS_DIR))
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("\n=== STEP 1: THE CONTRACT THE MODEL SEES ===")
    print(json.dumps(convert_to_openai_tool(search_docs), indent=2))

    print("\n=== STEP 2: THE MODEL PROPOSES A CALL ===")
    proposal = llm.bind_tools([search_docs]).invoke(
        [SystemMessage(SYSTEM_PROMPT),
         HumanMessage("What is prompt injection and how do we defend against it?")])
    print("content:", repr(proposal.content))
    print("tool_calls:", proposal.tool_calls)

    print("\n=== STEP 3: YOUR CODE EXECUTES IT ===")
    call = proposal.tool_calls[0]
    # Invoking a tool with the whole tool-call dict returns a ToolMessage
    # carrying the same tool_call_id; content_and_artifact fills .artifact.
    tool_message = search_docs.invoke(call)
    print("tool_call_id matches:", tool_message.tool_call_id == call["id"])
    print("content (sent to the model):", tool_message.content[:160], "...")
    print("artifact (kept for your code):", tool_message.artifact)

    print("\n=== STEP 4: ONE ROUND OF TOOLS IN A GRAPH ===")
    graph = build_single_round_graph(search_docs, llm)
    for question in ["What is prompt injection and how do we defend against it?",
                     "What is the capital of Canada?"]:
        result = graph.invoke({"messages": [HumanMessage(question)]})
        path = " -> ".join(m.type for m in result["messages"])
        print(f"\nQ: {question}\n  messages: {path}\n  A: {result['messages'][-1].content[:200]}")


if __name__ == "__main__":
    main()
