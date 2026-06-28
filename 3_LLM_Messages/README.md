# 3. LLM Messages

This tutorial shows how LangGraph can carry chat history through a graph.

## What You'll Learn

After this tutorial, you will be able to:

- Store conversation history in graph state using `add_messages`
- Send the full message history to an LLM inside a node
- Understand when to use manual `ChatState` vs LangGraph's built-in `MessagesState`

## Part 1 — Concept

Chatbots need memory. Not long-term memory yet — just the conversation so far.

In LangGraph, that conversation usually lives in a `messages` field inside the state.

```mermaid
flowchart TD
    A["Human message"] --> B["messages state"]
    B --> C["chatbot node"]
    C --> D["LLM sees full history"]
    D --> E["AI response"]
    E --> F["message history grows"]
```

The key idea: each node can return a new message, and LangGraph appends it to the existing message history.

The graph itself is simple:

```mermaid
flowchart LR
    START([START]) --> CHATBOT["chatbot node"]
    CHATBOT --> END([END])
```

### Built-In Message State

LangGraph provides `MessagesState`, a built-in state type for message history.

You can extend it when your graph needs extra fields:

```python
from langgraph.graph import MessagesState

class MyGraphState(MessagesState):
    turn_count: int
```

This means:

- `MessagesState` already gives you `messages`
- `MyGraphState` adds `turn_count`
- your graph state can now contain both

```python
{
    "messages": [...],
    "turn_count": 3
}
```

Conceptually, it is like this:

```python
class MyGraphState(TypedDict):
    messages: list
    turn_count: int
```

But `MessagesState` is special because it already handles LangGraph messages properly.

## Part 2 — Code Illustration

File:

```text
04_simple_chatbot.py
```

This example uses a manual `ChatState` on purpose — it shows the same pattern that `MessagesState` wraps for you: a `messages` field with the `add_messages` reducer attached.

The example starts with one human message:

```python
HumanMessage(content="What is RAG?")
```

The chatbot node sends the full conversation to the LLM:

```python
response = llm.invoke(state["messages"])
```

Then it returns only the new AI message:

```python
return {"messages": [response]}
```

`add_messages` appends the response to the existing history.

### Setup

Create a `.env` file in the repo root with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

Run it from the repo root:

```bash
python "3_LLM_Messages/04_simple_chatbot.py"
```

### Expected Output

You should see the initial human message, then a final conversation with two messages:

```text
HumanMessage: What is RAG?
AIMessage: RAG stands for Retrieval-Augmented Generation. ...
```

The exact AI reply will vary, but the structure is always: one human message in, one AI message appended.

## Code Explanation

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

This manually defines a message state. The `messages` field stores chat history, and `add_messages` appends new messages.

```python
def chatbot_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

This node receives the conversation history, calls the LLM, and returns the new AI message.

```python
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
```

This creates a one-node chatbot graph.

A more built-in style is:

```python
from langgraph.graph import MessagesState

class MyGraphState(MessagesState):
    turn_count: int
```

That keeps LangGraph's built-in message behavior and adds your own fields.

## What You Learned

- Conversation history lives in a `messages` field with the `add_messages` reducer
- Nodes return **only new messages**; LangGraph appends them to history
- `MessagesState` is a convenient shortcut when you do not need a custom state schema

## Next Step

Continue to [4. Conditional Edges](../4-Conditional%20Edges/README.md) to learn how a graph chooses different paths at runtime.
