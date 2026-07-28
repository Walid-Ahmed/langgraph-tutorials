# Long-Term Memory Diagrams

| Diagram | Purpose |
|---|---|
| `chat_update_memory_architecture.png` | supplied architecture sketch for the `chat → update_memory` pattern |
| `simple_cross_thread_memory_graph.png` | generated graph for `01_simple_cross_thread_memory.py` |
| `structured_cross_thread_memory_graph.png` | generated graph for `02_structured_cross_thread_memory.py` |

The architecture sketch explains both memory scopes:

- `MemorySaver` stores short-term messages within a `thread_id`;
- `InMemoryStore` shares selected facts across threads for a `user_id`.

`InMemoryStore` is cross-thread but process-local. It does not survive a Python
restart.
