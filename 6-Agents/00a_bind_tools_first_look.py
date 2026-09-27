# Run from the repository root:
#   python "6-Agents/00a_bind_tools_first_look.py"
#
# What it does:
#   Defines four smart-home tools, binds them to a chat model, sends four
#   requests, and prints what the model sends back. NO tool is ever executed:
#   there is no tool node, no loop and no code that runs the requested calls.
#
# What it demonstrates (Chapter 7 — Tools, §7.1, the book's first tool example):
#   - @tool turns a plain function into a tool (name + docstring + arguments)
#   - bind_tools only ADVERTISES the tools to the model; it never runs them
#   - the model's reply is an AIMessage whose tool_calls list says what it
#     WOULD like to run, with arguments it filled in itself
#   - one request can produce one call, several calls, or none at all
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env.
#
# Expected output (the model's wording may vary; the pattern should not):
#   Request 1 "Turn on the kitchen light"
#       -> text: '' ; tool_calls: [{'name': 'turn_on_light', 'args': {'room': 'kitchen'}, ...}]
#   Request 2 "Dim the bedroom lights to 30%"
#       -> text: '' ; tool_calls: [{'name': 'dim_lights', 'args': {'room': 'bedroom', 'brightness': 30}, ...}]
#   Request 3 "I'm cold and it's dark in the living room"
#       -> text: '' ; tool_calls: two entries, set_thermostat and turn_on_light
#   Request 4 "What's a good temperature for sleeping?"
#       -> text: an answer ; tool_calls: []
#   Final line: the device log is still empty — nothing was switched on.

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# If any tool body ever ran, it would write here. It never does in this file.
DEVICE_LOG: list[str] = []


@tool
def turn_on_light(room: str) -> str:
    """Turn on the lights in one room of the house, e.g. 'kitchen'."""
    DEVICE_LOG.append(f"light on: {room}")
    return f"The {room} light is on."


@tool
def set_thermostat(temperature: float, unit: Literal["C", "F"] = "C") -> str:
    """Set the home thermostat to a target temperature in Celsius (C) or Fahrenheit (F)."""
    DEVICE_LOG.append(f"thermostat: {temperature}{unit}")
    return f"Thermostat set to {temperature}°{unit}."


@tool
def dim_lights(room: str, brightness: int) -> str:
    """Dim the lights in one room to a brightness percentage from 0 (off) to 100 (full)."""
    DEVICE_LOG.append(f"dim {room}: {brightness}%")
    return f"The {room} lights are at {brightness}%."


@tool
def lock_door(door: str) -> str:
    """Lock one door of the house, e.g. 'front' or 'garage'."""
    DEVICE_LOG.append(f"locked: {door}")
    return f"The {door} door is locked."


tools = [turn_on_light, dim_lights, set_thermostat, lock_door]


REQUESTS = [
    "Turn on the kitchen light",
    "Dim the bedroom lights to 30%",
    "I'm cold and it's dark in the living room",
    "What's a good temperature for sleeping?",
]


def main() -> None:
    # Created here, not at import time, so 00b can reuse the tools without an API key.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)   # advertise the tools — nothing more

    for number, request in enumerate(REQUESTS, start=1):
        response = llm_with_tools.invoke(request)   # one call to the model, that's all
        print(f"\nRequest {number}: {request!r}")
        print("  text:      ", repr(response.content))
        print("  tool_calls:", response.tool_calls)   # printed only — never executed

    print(f"\nDevice log: {DEVICE_LOG}  <- still empty: no tool was executed")


if __name__ == "__main__":
    main()
