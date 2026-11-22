"""The two kinds of memory, and the helpers that read and write them.

    short-term memory = a CHECKPOINTER, keyed by thread_id
                        LangGraph saves the message list after every node.
                        New thread_id -> empty conversation.

    long-term memory  = a STORE, keyed by user_id
                        A key-value database shared by all of a user's
                        threads. This is where the profile lives.

What the store holds (namespace -> key -> value):

    ("memory",         user_id) / "user_details"  -> {"memory": "- Name: ..."}
    ("memory_history", user_id) / <iso timestamp> -> {"memory": ..., "thread_id": ...}
    ("threads",        user_id) / <thread_id>     -> {"title": ..., "created_at": ...}

The first row is the one that matters; the other two exist so the UI can show
a version history and a list of the user's conversations.

Every function below takes the ``store`` as its first argument. That is the
whole trick: the same helpers work with the in-memory store, the SQLite store,
and the store LangGraph injects into a graph node.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from ltm_chat.config import Settings

# store.search() returns only 10 items by default; we always want everything.
SEARCH_LIMIT = 1000

# The key the profile is stored under, inside the ("memory", user_id) namespace.
PROFILE_KEY = "user_details"


def profile_ns(user_id: str) -> tuple[str, str]:
    """Namespace holding the user's current profile."""
    return ("memory", user_id)


def history_ns(user_id: str) -> tuple[str, str]:
    """Namespace holding every past version of the profile."""
    return ("memory_history", user_id)


def threads_ns(user_id: str) -> tuple[str, str]:
    """Namespace holding the user's list of conversations."""
    return ("threads", user_id)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------- shapes
@dataclass(frozen=True)
class Profile:
    """What the assistant knows about one user.

    Used both for the current profile and for one entry of its history.
    """

    text: str
    saved_at: datetime
    thread_id: str | None = None  # the conversation that caused this version


@dataclass(frozen=True)
class Thread:
    """One conversation of one user."""

    thread_id: str
    title: str
    created_at: datetime


# ------------------------------------------------------- long-term: profile
def get_profile(store: BaseStore, user_id: str) -> Profile | None:
    """The user's current profile, or None if nothing is known yet."""
    item = store.get(profile_ns(user_id), PROFILE_KEY)
    if item is None:
        return None
    return Profile(text=item.value.get("memory", ""), saved_at=item.updated_at)


def get_profile_text(store: BaseStore, user_id: str) -> str | None:
    """Just the profile text - what the prompts need."""
    profile = get_profile(store, user_id)
    return profile.text if profile else None


def save_profile(
    store: BaseStore, user_id: str, text: str, thread_id: str | None = None
) -> None:
    """Overwrite the profile, and keep a copy in its history."""
    store.put(profile_ns(user_id), PROFILE_KEY, {"memory": text})
    # ISO timestamps sort alphabetically *and* chronologically, so using one
    # as the key means history comes back in order for free.
    store.put(
        history_ns(user_id),
        _now().isoformat(timespec="microseconds"),
        {"memory": text, "thread_id": thread_id},
    )


def get_history(store: BaseStore, user_id: str) -> list[Profile]:
    """Every saved version of the profile, oldest first."""
    items = store.search(history_ns(user_id), limit=SEARCH_LIMIT)
    versions = [
        Profile(
            text=i.value.get("memory", ""),
            saved_at=i.created_at,
            thread_id=i.value.get("thread_id"),
        )
        for i in items
    ]
    return sorted(versions, key=lambda p: p.saved_at)


def clear_profile(store: BaseStore, user_id: str) -> None:
    """Forget this user: delete the profile and its whole history."""
    store.delete(profile_ns(user_id), PROFILE_KEY)
    for item in store.search(history_ns(user_id), limit=SEARCH_LIMIT):
        store.delete(item.namespace, item.key)


# ------------------------------------------------------- long-term: threads
def register_thread(store: BaseStore, user_id: str, thread_id: str, title: str) -> None:
    """Remember that this thread belongs to this user.

    A checkpointer only knows about thread_ids, not which user owns them, so
    we keep that link in the store ourselves.
    """
    if store.get(threads_ns(user_id), thread_id) is not None:
        return  # already registered; keep the original title
    store.put(
        threads_ns(user_id),
        thread_id,
        {"title": title, "created_at": _now().isoformat()},
    )


def list_threads(store: BaseStore, user_id: str) -> list[Thread]:
    """The user's conversations, newest first."""
    items = store.search(threads_ns(user_id), limit=SEARCH_LIMIT)
    threads = [
        Thread(
            thread_id=i.key,
            title=i.value.get("title", i.key),
            created_at=datetime.fromisoformat(i.value["created_at"]),
        )
        for i in items
    ]
    return sorted(threads, key=lambda t: t.created_at, reverse=True)


def list_users(store: BaseStore) -> list[str]:
    """Every user_id the store has heard of."""
    users = {ns[1] for ns in store.list_namespaces(prefix=("memory",), max_depth=2)}
    users |= {ns[1] for ns in store.list_namespaces(prefix=("threads",), max_depth=2)}
    return sorted(users)


def dump_store(store: BaseStore) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Everything in the store, grouped by namespace (for the Store tab)."""
    return {
        ns: [i.dict() for i in store.search(ns, limit=SEARCH_LIMIT)]
        for ns in store.list_namespaces(limit=SEARCH_LIMIT)
    }


# ------------------------------------------------------------------ backends
@dataclass
class Backends:
    """A store (long-term) and a checkpointer (short-term), ready to use."""

    store: BaseStore
    checkpointer: BaseCheckpointSaver
    description: str
    connections: tuple[sqlite3.Connection, ...] = field(default=(), repr=False)

    def close(self) -> None:
        for conn in self.connections:
            conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Streamlit reruns the script on worker threads.
    # isolation_level=None: autocommit, like LangGraph's own helpers.
    return sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)


def create_backends(settings: Settings) -> Backends:
    """Build both kinds of memory from the settings.

    This is the only function that names a concrete store or checkpointer
    class, so adding Postgres later means adding one branch here.
    """
    if settings.backend == "memory":
        return Backends(
            store=InMemoryStore(),
            checkpointer=MemorySaver(),
            description="In memory (lost on restart)",
        )

    # Imported here so the "memory" backend works without the sqlite extra.
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore

    store_conn = _connect(settings.store_path)
    checkpoint_conn = _connect(settings.checkpoint_path)

    store = SqliteStore(store_conn)
    store.setup()  # creates the tables on first run, does nothing afterwards
    checkpointer = SqliteSaver(checkpoint_conn)
    checkpointer.setup()

    return Backends(
        store=store,
        checkpointer=checkpointer,
        description=f"SQLite ({settings.data_dir})",
        connections=(store_conn, checkpoint_conn),
    )
