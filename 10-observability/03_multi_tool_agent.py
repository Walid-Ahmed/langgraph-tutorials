# LangSmith multi-tool agent - the third observability example.
#
# This is Experiment 5 from langsmith_basics.ipynb, as a standalone script.
# LangChain's create_agent builds a compiled LangGraph tool loop. You provide
# two tools; the MODEL chooses which to call at runtime (and when to stop).
# You do not write ToolNode, bind_tools, or a router.
#
# Tools:
#   search_local_docs  - FAISS over data/llm_production_guide.txt (Exp 4 index)
#   google_search      - live web via Google Serper (needs SERPER_API_KEY)
#
# Why this matters for observability:
# - create_agent still returns a LangGraph, so LANGSMITH_* env vars trace the
#   whole loop: each LLM decision, each tool run, each follow-up LLM call.
# - One agent.invoke() is ONE trace. Nested runs show which tool the agent
#   picked. That is how you debug "wrong tool" or looping in production.
# - There is still no tracing code in the tools or the agent constructor.
#
# Before you run this (same LangSmith setup as 01, plus embeddings + Serper):
# 1. pip install -r requirements.txt
# 2. .env in THIS folder (copy .env.example):
#    OPENAI_API_KEY=...
#    LANGSMITH_TRACING=true
#    LANGSMITH_API_KEY=...
#    LANGSMITH_PROJECT=10-observability
#    SERPER_API_KEY=...          # optional; google_search degrades without it
#
# Run (from the 10-observability folder so data/ and .env resolve):
#    python 03_multi_tool_agent.py
#
# Expected result: three questions print a final answer and which tools ran.
# In smith.langchain.com, look for traces named:
#    Multi-Tool: local docs
#    Multi-Tool: web search
#    Multi-Tool: both
# Open a trace to see model -> tool -> model, not a fixed pipeline.

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

FOLDER = Path(__file__).resolve().parent
load_dotenv(FOLDER / ".env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def build_vectorstore() -> FAISS:
    """Chunk the local guide and embed it. Same pipeline as notebook Exp 4."""
    guide_path = FOLDER / "data" / "llm_production_guide.txt"
    if not guide_path.exists():
        raise SystemExit(f"Missing local guide: {guide_path}")

    raw_docs = TextLoader(str(guide_path), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(raw_docs)

    # Embeddings are extra OpenAI calls; they also show up in LangSmith.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def make_tools(vectorstore: FAISS):
    """
    You choose the toolbox (these two functions).
    The agent chooses which function to call for each question.
    Docstrings are the model's only API docs — keep them specific.
    """

    @tool
    def search_local_docs(query: str) -> str:
        """
        Search the internal LLM production guide.

        Use for RAG, security, evaluation, monitoring, prompt engineering,
        guardrails, and deployment topics in the local document.

        Call at most once per question, then answer the user.
        """
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant documents found."
        response = "\n\n".join(
            f"[Chunk {i + 1}]\n{doc.page_content[:700]}"
            for i, doc in enumerate(docs)
        )
        return response[:2500]

    @tool
    def google_search(query: str) -> str:
        """
        Search the live web for recent information.

        Use for current events, news, regulations, and recent AI developments.

        Call at most once per question, then answer the user.
        """
        if not os.getenv("SERPER_API_KEY"):
            return (
                "Web search unavailable: SERPER_API_KEY is missing. "
                "Answer from general knowledge or say you cannot fetch live news."
            )
        try:
            result = GoogleSerperAPIWrapper().run(query)
            if not result:
                return "No search results found."
            return str(result)[:2500]
        except Exception as exc:
            return f"Search failed: {exc}"

    return [search_local_docs, google_search]


SYSTEM_PROMPT = """
You are a research assistant.

You have two tools:

1. search_local_docs
   - Use for RAG, security, evaluation, monitoring,
     prompt engineering, guardrails, deployment.

2. google_search
   - Use for current events, news,
     regulations, recent AI developments.

Rules:
1. Call a tool ONLY if needed.
2. Never call the same tool more than once.
3. Maximum TWO total tool calls.
4. After receiving tool results, provide the final answer.
5. Do NOT continue searching if enough information exists.
6. Do NOT loop.
7. If one tool gives sufficient information, answer immediately.
"""

# Three questions chosen so the agent *should* pick different tools.
# The labels are expectations for you, not routing code — the model decides.
QUERIES = [
    (
        "What are LLM prompt injection attacks and how do we defend against them?",
        "Multi-Tool: local docs",
        "search_local_docs (expected, not enforced)",
    ),
    (
        "What are the latest AI regulations passed in 2025?",
        "Multi-Tool: web search",
        "google_search (expected, not enforced)",
    ),
    (
        "How does RAG work and what are the latest open-source RAG frameworks in 2025?",
        "Multi-Tool: both",
        "both tools (expected, not enforced)",
    ),
]


def tools_used(messages: list) -> list[str]:
    """Tool names that actually ran, in order, unique."""
    names = [
        msg.name for msg in messages if isinstance(msg, ToolMessage) and msg.name
    ]
    return list(dict.fromkeys(names))


def final_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to 10-observability/.env"
        )

    print("Building FAISS index from data/llm_production_guide.txt ...")
    vectorstore = build_vectorstore()
    tools = make_tools(vectorstore)

    # create_agent wires model -> tools -> model. Routing is the model's job.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    for question, run_name, hint in QUERIES:
        print("\n" + "=" * 72)
        print(f"Q: {question}")
        print(f"Hint (not used by the graph): {hint}")

        # recursion_limit caps the loop if the model keeps requesting tools.
        # run_name titles the LangSmith trace for this invoke.
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "run_name": run_name,
                "recursion_limit": 10,
            },
        )

        used = tools_used(result["messages"])
        answer = final_answer(result["messages"])
        print(f"Tools the agent actually called: {used or '(none)'}")
        print("\nANSWER:")
        print(answer[:800])

    print("\nOpen smith.langchain.com / project 10-observability")
    print("and compare the three traces named Multi-Tool: ...")


if __name__ == "__main__":
    main()
