# Run from the repository root:
#   python "6-Agents/00b_tool_call_ids.py"
#
# What it does:
#   Hands a ToolNode an AIMessage that asks for TWO tools at once (the reply the
#   model gave for "I'm cold and it's dark in the living room" in 00a), runs
#   them, and then matches every ToolMessage back to the request it answers by
#   its tool_call_id. No model and no API key are needed: the AIMessage is
#   written by hand so the ids are easy to read.
#
# What it demonstrates (Chapter 7 — Tools, §7.6):
#   - every entry in AIMessage.tool_calls has an "id"
#   - every ToolMessage carries tool_call_id = the id of the call it answers
#   - so a result can always be paired with its request, even when several
#     tools ran in the same turn
#   - this time the tools really run: ToolNode executes them (see DEVICE_LOG)
#
# Prerequisites: none beyond the repo requirements (no API key).
#
# Expected output (deterministic):
#   call_A: set_thermostat({'temperature': 22, 'unit': 'C'}) -> Thermostat set to 22.0°C.
#   call_B: turn_on_light({'room': 'living room'}) -> The living room light is on.
#   Device log: ['thermostat: 22.0C', 'light on: living room']

import importlib.util
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# Reuse the smart-home tools from 00a (file names starting with a digit
# cannot be imported with a plain `import` statement).
_spec = importlib.util.spec_from_file_location(
    "smart_home", Path(__file__).resolve().parent / "00a_bind_tools_first_look.py")
smart_home = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smart_home)

# A tiny graph whose only node runs tools: START -> tools -> END.
builder = StateGraph(MessagesState)
builder.add_node("tools", ToolNode(smart_home.tools))
builder.add_edge(START, "tools")
builder.add_edge("tools", END)
graph = builder.compile()

# The model's reply, written by hand: two tool calls in ONE message.
model_reply = AIMessage(content="", tool_calls=[
    {"name": "set_thermostat", "args": {"temperature": 22, "unit": "C"}, "id": "call_A"},
    {"name": "turn_on_light", "args": {"room": "living room"}, "id": "call_B"},
])


def main() -> None:
    messages = graph.invoke({"messages": [
        HumanMessage("I'm cold and it's dark in the living room"),
        model_reply,
    ]})["messages"]

    print("Messages after the tools node:")
    for msg in messages:
        print(f"  {type(msg).__name__:12} tool_call_id={getattr(msg, 'tool_call_id', None)!s:7} "
              f"content={msg.content!r}")

    # Index every request by its id, then look each result up by tool_call_id.
    requests = {call["id"]: call
                for msg in messages if isinstance(msg, AIMessage)
                for call in msg.tool_calls}

    print("\nEach result paired with the request it answers:")
    for msg in messages:
        if isinstance(msg, ToolMessage):
            call = requests[msg.tool_call_id]
            print(f"  {msg.tool_call_id}: {call['name']}({call['args']}) -> {msg.content}")

    print(f"\nDevice log: {smart_home.DEVICE_LOG}  <- this time the tools really ran")


if __name__ == "__main__":
    main()
