"""A chatbot that remembers you across conversations.

    short-term memory = checkpointer, scoped to a thread_id
    long-term memory  = store,        scoped to a user_id

Read the modules in this order:

    config.py    settings, and the models built from them
    prompts.py   the two prompts
    memory.py    the store: what it holds and how to read/write it
    graph.py     the two nodes and how they are wired
    service.py   MemoryChatService, the API the UI uses
"""

from ltm_chat.config import Settings, load_settings
from ltm_chat.service import MemoryChatService, TurnResult, new_thread_id

__all__ = [
    "MemoryChatService",
    "Settings",
    "TurnResult",
    "load_settings",
    "new_thread_id",
]
__version__ = "0.2.0"
