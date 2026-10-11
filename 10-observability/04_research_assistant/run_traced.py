# Research Assistant, traced - the last observability example in the sequence.
#
# This script runs the five-node Research Assistant graph from graph.py once and
# prints the final report. It contains no tracing code: the run is recorded in
# LangSmith only because the LANGSMITH_* environment variables are set.
#
# Why this matters:
# - A multi-node graph is where a trace pays off: the waterfall shows which node
#   the time went to (here, web_enricher and report_writer) and which nodes are
#   essentially free.
# - thread_id in the config does not change what the graph does (there is no
#   checkpointer); it is the key LangSmith uses to group traces into a thread.
#
# Before you run this:
# 1. Install this folder's dependencies (from the 10-observability folder):
#    pip install -r requirements.txt
# 2. Create a .env file in the 10-observability folder (copy .env.example):
#    OPENAI_API_KEY=your_openai_key
#    SERPER_API_KEY=your_serper_key            # from serper.dev
#    LANGSMITH_TRACING=true
#    LANGSMITH_API_KEY=your_langsmith_key      # from smith.langchain.com
#    LANGSMITH_PROJECT=10-observability
#
# Run it with (from the 10-observability folder, so .env and data/ are found):
#    python 04_research_assistant/run_traced.py
#
# Expected result: it prints a short, organized answer about LLM security risks
# (the exact text varies; it is an LLM), and one trace appears in the
# "10-observability" project with five node runs under the root: planner,
# document_reader, web_enricher, synthesizer, report_writer.

from dotenv import load_dotenv

# Load .env before importing graph: its models and tools read their keys at import time.
load_dotenv()

from graph import graph

config = {"configurable": {"thread_id": "demo-1"}}
inputs = {"query": "What are the top LLM security risks?", "session_id": "demo-1", "steps_taken": []}

result = graph.invoke(inputs, config)
print(result["final_report"])
