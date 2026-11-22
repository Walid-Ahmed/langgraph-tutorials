"""MemoryChatService - the one class the user interface talks to.

The Streamlit app never touches LangGraph, the store or the checkpointer
directly. It builds this service once and calls its methods:

    service = MemoryChatService.from_settings(load_settings())
    turn = service.chat(user_id="1", thread_id="abc123", text="Hi, I'm Sam")
    print(turn.reply, turn.memory_updated)

Everything below ``chat()`` is a read-only view used by the inspector panels.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage

from ltm_chat import memory
from ltm_chat.config import Settings, get_chat_model, get_memory_model, load_settings
from ltm_chat.graph import CHAT_NODE, build_graph

TITLE_LENGTH = 48


@dataclass(frozen=True)
class TurnResult:
    """What one user message did."""

    reply: str
    memory_before: str | None
    memory_after: str | None
    memory_updated: bool
    seconds: float


def new_thread_id() -> str:
    """A fresh thread_id - i.e. a brand new, empty conversation."""
    return uuid.uuid4().hex[:8]


def _title(text: str) -> str:
    """Shorten the first message into a label for the thread list."""
    one_line = " ".join(text.split())
    if len(one_line) <= TITLE_LENGTH:
        return one_line
    return one_line[: TITLE_LENGTH - 1] + "…"


class MemoryChatService:
    def __init__(self, graph, backends: memory.Backends) -> None:
        self.graph = graph
        self.backends = backends
        self.store = backends.store  # long-term memory

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        chat_model: BaseChatModel | None = None,
        memory_model: BaseChatModel | None = None,
    ) -> MemoryChatService:
        """Build everything from configuration.

        The two model arguments exist so you can pass fake models and run the
        whole graph without an API key.
        """
        settings = settings or load_settings()
        backends = memory.create_backends(settings)
        graph = build_graph(
            chat_model or get_chat_model(settings),
            memory_model or get_memory_model(settings),
            checkpointer=backends.checkpointer,
            store=backends.store,
        )
        return cls(graph, backends)

    def close(self) -> None:
        self.backends.close()

    # ---------------------------------------------------------------- one turn
    def chat(
        self,
        user_id: str,
        thread_id: str,
        text: str,
        on_token: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """Run one turn of the conversation.

        Pass ``on_token`` to receive the reply word by word as the model
        produces it; the full reply is in the returned TurnResult either way.
        """
        started = time.perf_counter()
        before = memory.get_profile_text(self.store, user_id)
        memory.register_thread(self.store, user_id, thread_id, _title(text))

        # thread_id picks the short-term memory, user_id the long-term memory.
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        final_state: dict[str, Any] = {}

        for mode, payload in self.graph.stream(
            {"messages": [HumanMessage(content=text)]},
            config,
            stream_mode=["messages", "values"],
        ):
            if mode == "values":
                final_state = payload
            elif on_token is not None:
                chunk, meta = payload
                # update_memory calls a model too - only stream the chat node.
                if meta.get("langgraph_node") == CHAT_NODE and isinstance(
                    chunk, AIMessageChunk
                ):
                    token = chunk.content if isinstance(chunk.content, str) else ""
                    if token:
                        on_token(token)

        return TurnResult(
            reply=str(final_state["messages"][-1].content),
            memory_before=before,
            memory_after=memory.get_profile_text(self.store, user_id),
            memory_updated=bool(final_state.get("memory_updated", False)),
            seconds=time.perf_counter() - started,
        )

    # ------------------------------------------------------ short-term views
    def get_messages(self, thread_id: str) -> list[BaseMessage]:
        """The messages the checkpointer holds for this thread."""
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list(state.values.get("messages", []))

    def checkpoint_count(self, thread_id: str) -> int:
        """How many checkpoints this thread has accumulated."""
        config = {"configurable": {"thread_id": thread_id}}
        return sum(1 for _ in self.graph.get_state_history(config))

    # ------------------------------------------------------- long-term views
    def get_profile(self, user_id: str) -> memory.Profile | None:
        return memory.get_profile(self.store, user_id)

    def get_history(self, user_id: str) -> list[memory.Profile]:
        return memory.get_history(self.store, user_id)

    def list_threads(self, user_id: str) -> list[memory.Thread]:
        return memory.list_threads(self.store, user_id)

    def list_users(self) -> list[str]:
        return memory.list_users(self.store)

    def forget_user(self, user_id: str) -> None:
        """Delete long-term memory. The conversations themselves stay."""
        memory.clear_profile(self.store, user_id)

    def store_dump(self) -> dict[tuple[str, ...], list[dict[str, Any]]]:
        return memory.dump_store(self.store)

    def graph_mermaid(self) -> str:
        """The compiled graph as a Mermaid diagram."""
        return self.graph.get_graph(xray=1).draw_mermaid()
