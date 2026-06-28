# 3. LLM Messages

This folder teaches how to use LangGraph with chat-style message history.

## Goal

Understand how a graph can keep conversation history and send that history to an LLM.

The key idea is that the state has a `messages` field. Each node can add new messages without deleting the old ones.

## Graph Plot

```mermaid
flowchart LR
    START([START]) --> CHATBOT["chatbot node"]
    CHATBOT --> END([END])
```

## Message Flow

```mermaid
flowchart TD
    A["HumanMessage"] --> B["messages state"]
    B --> C["chatbot node"]
    C --> D["LLM receives full history"]
    D --> E["AI response"]
    E --> F["add_messages appends response"]
```

## What The Example Does

File:

```text
04_simple_chatbot.py
```

The initial state contains one human message:

```python
HumanMessage(content="What is RAG?")
```

The chatbot node sends the full `messages` list to the LLM:

```python
response = llm.invoke(state["messages"])
```

Then it returns only the new AI message:

```python
return {"messages": [response]}
```

The `add_messages` reducer appends that response to the conversation history.

## Extending `MessagesState`

LangGraph provides a built-in `MessagesState` for conversation history. It already includes a special `messages` field that works correctly with LangGraph messages.

You can create your own custom state by inheriting from `MessagesState`:

```python
from langgraph.graph import MessagesState

class MyGraphState(MessagesState):
    turn_count: int
```

This means `MyGraphState` includes everything from `MessagesState`, plus one extra field called `turn_count`.

Conceptually, it is similar to writing:

```python
class MyGraphState(TypedDict):
    messages: list
    turn_count: int
```

But `MessagesState` is better for LangGraph chat workflows because it already handles message history properly.

A state using `MyGraphState` can look like this:

```python
{
    "messages": [...],
    "turn_count": 3
}
```

So yes: `MyGraphState` is a custom state class that extends LangGraph's built-in `MessagesState`.

## Setup

Create a local `.env` file before running this example:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Code Explanation

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

This defines state with a `messages` field. `add_messages` tells LangGraph to append new messages instead of replacing the list.

```python
def chatbot_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

This node receives the current conversation, calls the LLM, and returns the new AI message.

```python
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
```

This creates a one-node chatbot graph.

You can also use LangGraph's built-in `MessagesState` instead of manually defining the `messages` field:

```python
from langgraph.graph import MessagesState

class MyGraphState(MessagesState):
    turn_count: int
```

This keeps the built-in message handling and adds your own custom fields.
