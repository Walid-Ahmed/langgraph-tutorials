# A Chatbot That Remembers You

A chatbot built with **LangGraph** that remembers you *between* conversations,
with a Streamlit app that shows you exactly what it remembers and why.

The whole idea fits in one table:

| | Short-term memory | Long-term memory |
|---|---|---|
| LangGraph calls it a | **checkpointer** | **store** |
| Belongs to | one conversation (`thread_id`) | one person (`user_id`) |
| Holds | the messages of this chat | a profile of facts you stated |
| Start a new conversation | gone | still there |

What you get on screen:

```
┌─ sidebar ─────────┬─ chat ─────────────────────┬─ 🔍 memory inspector ──────┐
│ 👤 User ID    [1] │ 🧑‍💻 I am Sam, I build      │ 🧠 Long-term │ 💬 Short-term│
│                   │    TaskManager.            │ History │ Store │ Graph     │
│ 🧵 Conversation   │                            │                            │
│  ➕ New           │ 🤖 Hi Sam! Happy to help   │ - Name: Sam                │
│  ▶ I am Sam, ...  │    with TaskManager...     │ - Project: TaskManager     │
│                   │                            │ 🟢 learned this turn       │
│ 🎬 Try the demo   │ [ Message the assistant… ] │                            │
│ 🧹 Reset          │                            │ Messages 2 · Checkpoints 3 │
└───────────────────┴────────────────────────────┴────────────────────────────┘
```

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env            # macOS / Linux: cp .env.example .env
#   then open .env and paste your key into OPENAI_API_KEY

streamlit run app.py
```

Your browser opens at http://localhost:8501.

> There is no `pip install -e .` and no packaging step. `ltm_chat/` is a plain
> folder next to `app.py`, so Python finds it as long as you run commands from
> the project folder.

## See the point in 30 seconds

In the sidebar, open **🎬 Try the demo** and click the two buttons in order:

1. **Introduce yourself** — you say who you are. Watch the **🧠 Long-term** tab:
   the assistant writes a profile about you.
2. **New conversation, ask something** — this starts a *fresh* conversation.
   The **💬 Short-term** tab shows it is empty, yet the reply still says
   *"Sure, Evgeny... your TaskManager..."*, because the profile was pasted
   into the system prompt.

Then stop the app, start it again, and the profile is still there — it lives in
a SQLite file under `data/`.

## The code

Read the files in this order. Each one does a single job.

```
app.py              the Streamlit UI (drawing only, no memory logic)

ltm_chat/
  config.py         settings from .env, and the chat models built from them
  prompts.py        the two prompts: one answers, one updates the profile
  memory.py         the store: what it holds, and get/save/clear helpers
  graph.py          the two nodes and how they are wired together
  service.py        MemoryChatService - the one class the UI talks to
```

The graph is a straight line with two steps:

```
START  ->  chat  ->  update_memory  ->  END
```

* **`chat`** reads your profile from the store, puts it in the system prompt,
  and answers you.
* **`update_memory`** reads your profile plus the whole conversation, asks a
  second model for an updated profile, and saves it — or answers `NO_UPDATE`
  and changes nothing.

What the store holds:

```
("memory",         user_id) / "user_details"   ->  the current profile
("memory_history", user_id) / <timestamp>      ->  every past version
("threads",        user_id) / <thread_id>      ->  this user's conversations
```

Only the first line is essential. The other two exist so the app can show you
a history and a list of conversations.

## Settings

All optional except the key. Put them in `.env`.

| Variable | Default | What it does |
|---|---|---|
| `OPENAI_API_KEY` | — | **required** |
| `OPENAI_BASE_URL` | — | point at any OpenAI-compatible server (Ollama, OpenRouter, vLLM) |
| `LTM_CHAT_MODEL` | `gpt-4o-mini` | the model that answers you |
| `LTM_MEMORY_MODEL` | same as above | the model that updates the profile |
| `LTM_CHAT_TEMPERATURE` | `0.7` | creativity of the answers |
| `LTM_BACKEND` | `sqlite` | `sqlite` keeps memory on disk, `memory` forgets on exit |
| `LTM_DATA_DIR` | `./data` | where the SQLite files go |

Run it without saving anything to disk:

```bash
# Windows PowerShell
$env:LTM_BACKEND="memory"; streamlit run app.py

# macOS / Linux
LTM_BACKEND=memory streamlit run app.py
```

## Things to try

* Type a different **User ID** in the sidebar and say something about yourself.
  Switch back and forth: the two profiles never mix.
* Say something that contradicts what you said before ("actually I use Go
  now"). Open the **History** tab and watch the profile get rewritten.
* Say something with no facts in it ("thanks!"). The **🧠 Long-term** tab
  reports that nothing was learned — that is the `NO_UPDATE` reply.
* Open the **Store** tab to see the raw key-value data behind all of it.

## Exercises

1. **Remember more than a profile.** Add a `("summaries", user_id)` namespace
   and a third node that summarises a finished conversation.
2. **Use Postgres instead of SQLite.** Add one branch to `create_backends()` in
   `ltm_chat/memory.py` — it is the only function that names a concrete store
   class, so nothing else has to change.
3. **Let the user correct the profile.** Add a text box to the Long-term tab
   that calls `memory.save_profile()` directly.
4. **Stop trusting the model.** `update_memory` saves whatever text comes back.
   Make it reject a reply that is suspiciously long or that drops facts.

See [docs/DESIGN.md](docs/DESIGN.md) for the architecture: C4 diagrams, the data
model, every design decision and its trade-off, and the extension paths. The
diagrams there are inline Mermaid, so they render on GitHub and in the VS Code
Markdown preview without any image files.
