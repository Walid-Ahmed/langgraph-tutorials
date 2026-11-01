# The agentic RAG chatbot: a ReAct agent that decides when to search the knowledge
# base, traced in LangSmith, with reference-free online scores on every live answer.
#
# Why this matters:
# - create_agent builds a LangGraph graph (the model node and a tool node in a
#   loop), so this is the Chapter 7 agent loop with a vector store behind one tool.
# - The chat UI and the offline evaluation both call ask_with_tools, so the
#   evaluation measures the same code path users hit.
# - Online scoring runs in a background thread: the user gets the answer at once,
#   and the scores land on the run's trace a few seconds later as LangSmith feedback.
#
# Prerequisites: OPENAI_API_KEY, LANGSMITH_API_KEY and LANGSMITH_TRACING=true in .env,
# and an ingested knowledge base (python -m src.pipeline). Imported, not run.

import threading
import uuid
from dataclasses import dataclass, field
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_openai import ChatOpenAI
from langsmith import Client

from src import config
from src.tools import ALL_TOOLS, SOURCE_TAG

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


@dataclass
class AgentReply:
    answer: str
    tools: list[str]                                    # tool names, in call order
    sources: list[str] = field(default_factory=list)    # files the searches returned, in rank order
    contexts: list[str] = field(default_factory=list)   # the retrieved chunk texts
    run_id: str = ""                                     # the trace this answer belongs to


@lru_cache(maxsize=1)
def get_agent():
    model = ChatOpenAI(model=config.AGENT_MODEL, temperature=0)
    return create_agent(model, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


def _retrieved(messages) -> tuple[list[str], list[str]]:
    """Read back what the knowledge-base searches returned: (sources, chunk texts)."""
    sources, contexts = [], []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_knowledge_base":
            for block in message.text.split("\n\n---\n\n"):
                if block.startswith(SOURCE_TAG):
                    header, _, text = block.partition("\n")
                    sources.append(header[len(SOURCE_TAG):-1])
                    contexts.append(text)
    return sources, contexts


_scoring_threads: list[threading.Thread] = []


def wait_for_online_scores(timeout: float = 120) -> None:
    """Block until the background scoring threads finish; a script calls this before it exits."""
    for thread in list(_scoring_threads):
        thread.join(timeout)


def _score_online(run_id: str, question: str, answer: str) -> None:
    """Reference-free checks for live traffic, where no golden answer exists."""
    from evals.metrics.judges import relevance_judge   # the judge loads only when used
    from evals.metrics.safety_metrics import bias_fairness_score, toxicity_score

    wait_for_all_tracers()  # the run must reach LangSmith before feedback can attach to it
    client = Client()
    for result in (toxicity_score(answer), bias_fairness_score(answer)):
        client.create_feedback(run_id, key=result.name, score=result.score, comment=result.note)
    try:
        grade = relevance_judge(question, answer)
        client.create_feedback(run_id, key="answer_relevancy", score=grade.score, comment=grade.note)
    except Exception as exc:  # a failed judge call must not hide the other scores
        client.create_feedback(run_id, key="answer_relevancy_error", comment=str(exc))


def ask_with_tools(question: str, chat_history: list[dict] | None = None,
                   session_id: str | None = None, online_eval: bool = True) -> AgentReply:
    """Ask the agent one question; report the answer, the tools it called and what it retrieved.

    session_id groups a conversation's questions into one LangSmith thread.
    online_eval is turned off by evals/run_eval.py, which grades against the golden set instead.
    """
    run_id = uuid.uuid4()   # chosen up front so feedback can find this run later
    messages = [*(chat_history or []), {"role": "user", "content": question}]
    result = get_agent().invoke({"messages": messages}, config={
        "run_id": run_id,
        "run_name": "chatbot",
        "tags": ["chatbot"],
        "metadata": {"session_id": session_id} if session_id else {},
    })
    new_messages = result["messages"][len(messages):]   # only this turn, not the history
    sources, contexts = _retrieved(new_messages)
    reply = AgentReply(
        answer=result["messages"][-1].text,
        tools=[m.name or "" for m in new_messages if isinstance(m, ToolMessage)],
        sources=sources, contexts=contexts, run_id=str(run_id),
    )
    if online_eval:
        thread = threading.Thread(target=_score_online, args=(reply.run_id, question, reply.answer),
                                  daemon=True)  # the user never waits for the scores
        thread.start()
        _scoring_threads.append(thread)
    return reply
