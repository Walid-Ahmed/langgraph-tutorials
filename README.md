# LangGraph Tutorials

A beginner-friendly tutorial repo for learning LangGraph one concept at a time.

This repo is meant to feel like a guided path, not a code dump. Each folder introduces one idea, explains why it matters, then uses a small Python file to make the idea concrete.

## Prerequisites

- Python 3.10 or newer
- Basic Python (functions, dictionaries, classes)
- An OpenAI API key for LLM examples in tutorials 3, 5, 6, 7, 8, 9, and some exercise solutions

For deeper reference, see the [official LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview).

This repo intentionally follows the official LangGraph mental model: define **state**, run **nodes**, connect them with **edges**, then compile the graph into something you can invoke. The examples are small so the idea is visible before the code becomes realistic.

## Part 1 — Core Tutorial Roadmap

LangGraph lets you build workflows as graphs. A graph is made of three main pieces:

| Piece | Meaning | Simple Way To Think About It |
|---|---|---|
| State | Data moving through the graph | The backpack your workflow carries |
| Node | A function that does work | A step in the workflow |
| Edge | A connection between nodes | The road to the next step |

```mermaid
flowchart LR
    START([START]) --> NODE["node function"]
    NODE --> UPDATE["state update"]
    UPDATE --> END([END])
```

The learning path builds up slowly:

```mermaid
flowchart TD
    A["1. Basic Graph"] --> B["2. Reducers"]
    B --> C["3. LLM Messages"]
    C --> D["4. Conditional Edges"]
    D --> E["5. Workflows"]
    E --> F["6. Agents"]
    F --> G["7. Checkpointing"]
    G --> H["8. Long-Term Memory"]
    H --> I["9. Email Assistant"]
    I -.-> J["Exercise Solutions"]
```

Each tutorial follows the same rhythm:

1. the concept and the problem it solves, in plain language with an intuition-building analogy
2. the architecture of the example — a diagram and a table of what each stage reads and writes
3. code highlights explaining *why* the important lines are designed the way they are
4. a step-by-step execution walkthrough showing how state evolves
5. exercises (with solutions in `Exercise-Solutions/`) and key takeaways

## Folder Guide

| Folder | Tutorial Focus | Why It Matters |
|---|---|---|
| `1-Langgraph basics/` | Build the smallest possible graph | Learn the core shape: state, node, edge, compile, invoke |
| `2-Reducer/` | Compare state updates with and without reducers | Understand how LangGraph preserves or combines state |
| `3_LLM_Messages/` | Store chat history in graph state | Learn how LLM conversations fit into LangGraph |
| `4-Conditional Edges/` | Route to different nodes | Learn how graphs make decisions |
| `5-Workflows/` | Workflow patterns | Larger LLM designs such as routing, parallel work, orchestration, and evaluation loops |
| `6-Agents/` | Agent patterns | Compare manual routers, `Command`, `ToolNode`, and high-level ReAct-style agents |
| `7-Checkpointing/` | Persist state across runs | Learn thread memory with `MemorySaver`, durable checkpoints with `PostgresSaver`, and how this differs from long-term memory |
| `8-Long-Term-Memory/` | Share selected memory across conversations | Learn Store namespaces, `user_id`, `InMemoryStore`, and the path to `PostgresStore` |
| `9-Email-Assistant/` | Build a complete assistant gradually | Apply routing, tools, and short-term, semantic, episodic, and procedural memory |
| `Exercise-Solutions/` | Practice solutions | Runnable answers for the exercises at the end of each tutorial |


## Memory Scopes in This Repo

LangGraph uses the word "memory" in a few related ways. This repo separates them so the ideas do not blur together:

| Memory Type | Scope | Stored In | Survives Python Restart? | Covered In |
|---|---|---|---|---|
| No memory | one isolated invoke | nowhere | no | `7-Checkpointing/02-memory-saver/00_no_memory.py` |
| Manual history | caller-managed conversation | your Python variable / app code | only if your app saves it | `7-Checkpointing/02-memory-saver/02_manual_history.py` |
| `MemorySaver` | one LangGraph thread | Python process memory | no | `7-Checkpointing/02-memory-saver/01_memory_saver.py` |
| `PostgresSaver` | many durable LangGraph threads, each keyed by `thread_id` | PostgreSQL checkpoint tables | yes | `7-Checkpointing/06-postgres-saver/` |
| `InMemoryStore` | cross-thread user or app facts | Python process memory | no | `8-Long-Term-Memory/` |
| `PostgresStore` | durable cross-thread user or app facts | PostgreSQL store tables | yes | `8-Long-Term-Memory/03-postgres-store/` |

The most important distinction:

```mermaid
flowchart TD
    MEMORY["LangGraph memory"]

    MEMORY --> SHORT["Short-term memory<br/>one conversation"]
    SHORT --> THREAD["identified by thread_id"]
    THREAD --> MS["MemorySaver<br/>Python process only"]
    THREAD --> PS["PostgresSaver<br/>durable checkpoints"]

    MEMORY --> LONG["Long-term memory<br/>shared across conversations"]
    LONG --> USER["identified by user_id<br/>inside a Store namespace"]
    USER --> IMS["InMemoryStore<br/>Python process only"]
    USER --> PGS["PostgresStore<br/>durable user facts"]

    MEMORY --> MANUAL["Manual history<br/>caller stores and resends messages"]
```

```text
Saver = checkpoints graph state and messages by thread_id
Store = saves selected user or application facts by namespace + key
```

`PostgresSaver` can hold many conversations, but each one is still separate by `thread_id`. It remembers this thread:

```text
thread_id = "chat_session_walid"
→ messages and graph state for that conversation
```

Long-term memory is different. It is usually keyed by a stable user or application id and can be reused across many threads:

```text
user_id = "walid"
→ preferences, profile, durable facts
```

So persistence alone does not mean "long-term memory." `PostgresSaver` persists checkpoints; `Store` is where cross-conversation facts belong.

### Long-term memory content types

Memory scope describes **where and how long** information is available. Memory
type describes **what the information means**:

| Content type | Meaning | Current coverage |
|---|---|---|
| Semantic | facts about users, people, places, and things | implemented with Store and LangMem in tutorials 8 and 9 |
| Episodic | past actions and outcomes used as examples | implemented as retrieved, human-corrected triage examples in tutorial 9 |
| Procedural | instructions that control behavior | implemented as per-user, feedback-optimized stored instructions in tutorial 9 |

### Long-term memory examples

Run the new tutorial in this order:

1. [`00_store_basics.py`](8-Long-Term-Memory/00_store_basics.py) — Store hello world with `put`, `get`, and `search`; no LLM or API key.
2. [`01_simple_cross_thread_memory.py`](8-Long-Term-Memory/01_simple_cross_thread_memory.py) — the simplest complete chatbot using `MemorySaver` plus `InMemoryStore`.
3. [`02_structured_cross_thread_memory.py`](8-Long-Term-Memory/02_structured_cross_thread_memory.py) — structured extraction, safer merging, and user isolation.
4. [`03-postgres-store/`](8-Long-Term-Memory/03-postgres-store/) — save a profile in one process and reload it from PostgreSQL in another.
5. [`9-Email-Assistant/`](9-Email-Assistant/) — apply the memory concepts in a complete email assistant built gradually.

The chatbot examples make two LLM calls per turn: `chat` produces the user-facing response, then `update_memory` extracts and saves user facts. See the [long-term memory tutorial](8-Long-Term-Memory/) for diagrams and a complete walkthrough.

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For LLM examples, create a local `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_api_key_here
```

For the tool-calling agent, optionally add API keys for live weather and web search:

```bash
OPENWEATHER_API_KEY=your_openweather_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## Suggested Order

Read and run the folders in order:

1. [`1-Langgraph basics/`](1-Langgraph%20basics/)
2. [`2-Reducer/`](2-Reducer/)
3. [`3_LLM_Messages/`](3_LLM_Messages/)
4. [`4-Conditional Edges/`](4-Conditional%20Edges/)
5. [`5-Workflows/`](5-Workflows/)
6. [`6-Agents/`](6-Agents/)
7. [`7-Checkpointing/`](7-Checkpointing/)
8. [`8-Long-Term-Memory/`](8-Long-Term-Memory/)
9. [`9-Email-Assistant/`](9-Email-Assistant/)

Use [`Exercise-Solutions/`](Exercise-Solutions/) after trying the exercises yourself.

Each tutorial folder has its own README that works like a mini lesson.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'langgraph'` | Activate the virtual environment and run `pip install -r requirements.txt` |
| `OpenAI` authentication error in tutorials 3, 5, 6, 7, or 8 | Check that `.env` exists in the repo root and contains a valid `OPENAI_API_KEY` |
| Run commands fail with "file not found" | Run commands from the repo root, not from inside a tutorial folder |

## Official References Used

These tutorials are enriched from the official LangChain and LangGraph docs, then simplified into beginner examples:

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)

## Getting Started

Tutorial 1 walks through the core graph pattern step by step. Once you understand that shape, the rest of the series builds on it. Start with [`1-Langgraph basics/README.md`](1-Langgraph%20basics/README.md).
