import sys
from pathlib import Path
from typing import TypedDict, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import Literal

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util import plot_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")


# ---------------------------------------------------------
# 1. Structured output schema for the router
#
# The LLM must return one of three literals so the router
# has a reliable label to branch on — no free-text parsing.
# ---------------------------------------------------------

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        description="The next step in the routing process"
    )


router = llm.with_structured_output(Route)


# ---------------------------------------------------------
# 2. State
#
# - input   : the user's request, read by the router and workers
# - decision: the route label written by the router node
# - output  : the final text written by whichever worker runs
# ---------------------------------------------------------

class State(TypedDict):
    input: str
    decision: str
    output: str


# ---------------------------------------------------------
# 3. Worker nodes
#
# Each node handles one content type. They share the same
# structure — invoke the LLM with the user input and write
# the result to output.
# ---------------------------------------------------------

def llm_call_1(state: State):
    """Write a story"""
    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_2(state: State):
    """Write a joke"""
    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_3(state: State):
    """Write a poem"""
    result = llm.invoke(state["input"])
    return {"output": result.content}


# ---------------------------------------------------------
# 4. Router node
#
# Uses structured output to classify the input and write a
# reliable route label ("story", "joke", or "poem") to state.
# ---------------------------------------------------------

def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    decision = cast(Route, router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke, or poem based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    ))
    return {"decision": decision.step}


# ---------------------------------------------------------
# 5. Conditional edge
#
# Reads the decision written by the router node and returns
# the name of the worker node to visit next.
# ---------------------------------------------------------

def route_decision(state: State):
    """Route to the correct worker based on the router's decision"""

    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"


# ---------------------------------------------------------
# 6. Build graph
# ---------------------------------------------------------

router_builder = StateGraph(State)

router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # label returned by route_decision : next node to visit
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

router_workflow = router_builder.compile()

plot_graph(router_workflow)

# ---------------------------------------------------------
# 7. Run
# ---------------------------------------------------------

state = router_workflow.invoke({"input": "Write me a joke about cats"})  # type: ignore[arg-type]
print(state["output"])
