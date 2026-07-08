# Same chatbot as 02_no_memory.py, but compiled WITH a MemorySaver
# checkpointer and a fixed thread_id. Now a second invoke() on the same
# thread remembers the first turn, proving the checkpointer is what
# provides conversational memory.

from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver

client = OpenAI()

# LangChain messages use type "human"/"ai", but the OpenAI API expects "user"/"assistant".
ROLE_MAP = {"human": "user", "ai": "assistant"}


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: State):
    messages = [
        {"role": ROLE_MAP.get(m.type, m.type), "content": m.content}
        for m in state["messages"]
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=256,
    )
    return {"messages": [{"role": "assistant", "content": response.choices[0].message.content}]}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)  # attach checkpointer

config = {"configurable": {"thread_id": "walid-session"}}  # thread ties runs together

# Run 1 — introduce yourself
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Walid"}]}, config)

# Run 2 — graph REMEMBERS run 1 via thread_id
result = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)
print("Bot:", result["messages"][-1].content)
# → "Your name is Walid!"

# Inspect what the checkpointer actually stored for this thread
state = graph.get_state(config)
print("Stored state:", state.values)
