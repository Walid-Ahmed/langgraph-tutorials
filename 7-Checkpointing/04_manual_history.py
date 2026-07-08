from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from openai import OpenAI

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

graph = builder.compile()  # no checkpointer needed

# Run 1 — introduce yourself
result = graph.invoke({
    "messages": [{"role": "user", "content": "Hi, my name is Walid"}]
})
print("Bot:", result["messages"][-1].content)

# Run 2 — pass ALL previous messages so it remembers
result = graph.invoke({
    "messages": result["messages"] + [{"role": "user", "content": "What is my name?"}]
})
print("Bot:", result["messages"][-1].content)

# Run 3 — keep growing the history
result = graph.invoke({
    "messages": result["messages"] + [{"role": "user", "content": "And say goodbye to me by name!"}]
})
print("Bot:", result["messages"][-1].content)
