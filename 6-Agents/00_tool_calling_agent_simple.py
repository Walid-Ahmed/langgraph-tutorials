# A tool-calling agent built by hand: the LLM is bound to three arithmetic
# tools, and a manually-written tool_node executes whichever tools the model
# requests. Shows what LangGraph's prebuilt ToolNode does internally,
# looping llm_call -> tool_node -> llm_call until the model stops calling tools.

import sys
from pathlib import Path


from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph import MessagesState

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------
# 1. Tools
#
# Simple math tools. Each one is a plain Python function
# decorated with @tool so the LLM can request them by name.
# ---------------------------------------------------------

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b


tools = [add, multiply, divide]

# Lookup dict so the tool node can find tools by name from the LLM response.
tools_by_name = {t.name: t for t in tools}

# bind_tools() tells the model which tools exist and what arguments they take.
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------
# 2. LLM Node
#
# Prepends a system prompt to every call so the model knows
# its role. The system message + existing history are passed
# together so the model has full context.
# ---------------------------------------------------------

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ]
    }


# ---------------------------------------------------------
# 3. Tool Node
#
# Manually iterates over every tool call in the last message,
# runs the matching Python function, and wraps each result
# in a ToolMessage so the LLM can read it on the next pass.
#
# This is what ToolNode (used in 01_tool_calling_agent.py)
# does internally — written out here so the loop is visible.
# ---------------------------------------------------------

def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    last_message = state["messages"][-1]
    for tool_call in getattr(last_message, "tool_calls", []):
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


# ---------------------------------------------------------
# 4. Router
#
# Checks whether the last LLM message contains tool calls.
# If yes, route to tool_node. If no, the answer is ready.
# ---------------------------------------------------------

def should_continue(state: MessagesState) -> str:
    """Route to tool_node if the LLM requested tools, otherwise end"""

    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


# ---------------------------------------------------------
# 5. Build Graph
#
# Agent loop:  llm_call → should_continue → tool_node → llm_call → ...
# The LLM decides how many iterations happen.
# ---------------------------------------------------------

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

plot_graph(agent)

# ---------------------------------------------------------
# 6. Run
# ---------------------------------------------------------

messages = [HumanMessage(content="Add 3 and 4. Then multiply the result by 2.")]
result = agent.invoke({"messages": messages})  # type: ignore[arg-type]
for m in result["messages"]:
    m.pretty_print()
