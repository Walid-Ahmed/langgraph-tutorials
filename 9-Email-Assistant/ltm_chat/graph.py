"""The graph: two nodes, one straight line.

    START -> chat -> update_memory -> END

``chat``           reads the profile out of the store, puts it in the system
                   prompt, and answers the user.
``update_memory``  reads the profile and the whole conversation, asks a second
                   model for a merged profile, and writes it back.

LangGraph hands each node its ``config`` and the ``store`` automatically,
just because of the parameter names. A node never creates a store itself.

The two ``make_*_node`` functions take a model and return the node function.
That little bit of indirection is what lets you build the same graph with a
real model or a fake one.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore

from ltm_chat import memory
from ltm_chat.prompts import NO_UPDATE, chat_prompt, memory_prompt

CHAT_NODE = "chat"
UPDATE_MEMORY_NODE = "update_memory"


class ChatState(MessagesState):
    """The graph's state.

    ``MessagesState`` already gives us ``messages``, which grows instead of
    being replaced. We add one flag so the UI can say whether the last turn
    changed long-term memory, without comparing the store itself.
    """

    memory_updated: bool


def get_user_id(config: RunnableConfig) -> str:
    """user_id comes from the config, exactly like thread_id does."""
    try:
        return str(config["configurable"]["user_id"])
    except KeyError as exc:
        raise KeyError(
            "Pass config={'configurable': {'thread_id': ..., 'user_id': ...}}"
        ) from exc


def get_thread_id(config: RunnableConfig) -> str | None:
    return config.get("configurable", {}).get("thread_id")


def is_no_update(text: str) -> bool:
    """True when the memory model says nothing new was learned."""
    cleaned = text.strip().strip("`").strip()
    return cleaned == "" or cleaned.upper() == NO_UPDATE


def make_chat_node(model: BaseChatModel):
    """Build the node that answers the user."""

    def chat(state: ChatState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
        user_id = get_user_id(config)
        # Long-term memory goes into the system prompt. This one line is why a
        # brand new thread still knows who the user is.
        profile = memory.get_profile_text(store, user_id)
        system = SystemMessage(content=chat_prompt(profile))

        reply = model.invoke([system, *state["messages"]])
        return {"messages": [reply]}

    return chat


def make_update_memory_node(model: BaseChatModel):
    """Build the node that rewrites the profile after the answer."""

    def update_memory(
        state: ChatState, config: RunnableConfig, store: BaseStore
    ) -> dict[str, Any]:
        user_id = get_user_id(config)
        current = memory.get_profile_text(store, user_id)

        system = SystemMessage(content=memory_prompt(current))
        result = model.invoke([system, *state["messages"]])
        new_text = str(result.content).strip()

        # Nothing new, or the same thing again: leave the store alone.
        if is_no_update(new_text) or new_text == (current or "").strip():
            return {"memory_updated": False}

        memory.save_profile(store, user_id, new_text, thread_id=get_thread_id(config))
        return {"memory_updated": True}

    return update_memory


def build_graph(
    chat_model: BaseChatModel,
    memory_model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore,
):
    """Wire the nodes together and compile, with both kinds of memory attached.

    ``checkpointer`` is short-term memory, ``store`` is long-term memory.
    """
    builder = StateGraph(ChatState)
    builder.add_node(CHAT_NODE, make_chat_node(chat_model))
    builder.add_node(UPDATE_MEMORY_NODE, make_update_memory_node(memory_model))

    builder.add_edge(START, CHAT_NODE)
    builder.add_edge(CHAT_NODE, UPDATE_MEMORY_NODE)
    builder.add_edge(UPDATE_MEMORY_NODE, END)

    return builder.compile(checkpointer=checkpointer, store=store)
