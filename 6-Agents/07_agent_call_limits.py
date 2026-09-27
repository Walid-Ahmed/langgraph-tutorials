# Run from the repository root:
#   python "6-Agents/07_agent_call_limits.py"
#
# What it does:
#   Gives create_agent a tool that never finishes ("still pending, check again")
#   and a request to keep checking until it is done — so, left alone, the agent
#   would loop until the recursion limit. Runs it twice, each time with one
#   built-in limit:
#     1. ModelCallLimitMiddleware(run_limit=3)  -> at most 3 model calls
#     2. ToolCallLimitMiddleware(tool_name="check_status", run_limit=2,
#                                exit_behavior="end") -> at most 2 tool runs
#   and prints how many times the tool ran and the last message.
#
# What it demonstrates (Chapter 8 — Agents, §8.6 "What stops the loop?"):
#   - you control how many laps a create_agent agent may take, without writing
#     the loop yourself, by passing middleware=[...]
#   - a limit ends the run cleanly (no exception); the last message says why
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (message wording may vary; the counts should not):
#   Model-call limit (3): check_status ran 3 time(s)
#     last message: Model call limits exceeded: run limit (3/3)
#   Tool-call limit (2): check_status ran 2 time(s)
#     last message: 'check_status' tool call limit reached: ...

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


@tool
def check_status() -> str:
    """Check whether the nightly backup job has finished."""
    return "Still pending. Check again."   # never finishes: a runaway loop on purpose


REQUEST = "Keep checking the backup status, one check at a time, until it says done."


def run(label: str, middleware) -> None:
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[check_status],
        middleware=[middleware],           # the only difference between the two runs
    )
    result = agent.invoke({"messages": [HumanMessage(REQUEST)]})
    ran = sum(1 for m in result["messages"]
              if isinstance(m, ToolMessage) and m.status != "error")
    print(f"{label}: check_status ran {ran} time(s)")
    print(f"  last message: {result['messages'][-1].content}")


def main() -> None:
    run("Model-call limit (3)", ModelCallLimitMiddleware(run_limit=3))
    run("Tool-call limit (2)",
        ToolCallLimitMiddleware(tool_name="check_status", run_limit=2, exit_behavior="end"))


if __name__ == "__main__":
    main()
