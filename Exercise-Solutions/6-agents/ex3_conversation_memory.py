# Run from the repository root:
#   python "Exercise-Solutions/6-agents/ex3_conversation_memory.py"
#
# Exercise 7.4 — Multi-turn by hand (Appendix B, Listing B.18)
#
# What it shows: without a checkpointer the caller carries the history.
# We save first["messages"], append the follow-up question, and pass the
# whole list back, so the agent can answer "What did I just ask you?".
# Chapter 9's checkpointer does the same thing for you, keyed by thread_id.
# Needs OPENAI_API_KEY in the repository-root .env file.
#
# Expected output (representative):
#   You asked me to add 3 and 4, then multiply the result by 2 — which came to 14.

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# The Chapter 7 arithmetic agent (6-Agents/03_prebuilt_react_agent.py).
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add, multiply],
    system_prompt="You are an arithmetic assistant. "
                  "Always use the provided tools for calculations.",
)


def main() -> None:
    first = agent.invoke({"messages": [HumanMessage("Add 3 and 4, then multiply by 2.")]})

    history = first["messages"] + [HumanMessage("What did I just ask you?")]
    second = agent.invoke({"messages": history})
    print(second["messages"][-1].content)


if __name__ == "__main__":
    main()
