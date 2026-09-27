# Run from the repository root:
#   python "8-Long-Term-Memory/05_episodic_memory.py"
#
# What it does:
#   A one-node graph answers "explain <concept>" requests. When the user approves
#   (or edits) an answer, that exchange is saved as an EPISODE. For later,
#   similar requests the node retrieves the closest episodes by embedding
#   similarity and shows them to the model as few-shot examples, so new answers
#   follow the format the user liked, without changing the instructions.
#
# What it demonstrates (Chapter 11 — episodic memory):
#   - episodic memory = specific past experiences and their outcome
#   - background updates: episodes are written by the app after a human approves
#   - store.search(namespace, query=...) to find similar past episodes
#   - few-shot prompting as the way the model uses them
#   - per-user isolation of episodes
#
# Prerequisites:
#   OPENAI_API_KEY in the repo-root .env (chat model + embeddings).
#
# Expected output (wording varies):
#   Before -> 0 episodes retrieved, a generic explanation of recursion
#   After  -> 1 episode retrieved; the explanation of dynamic programming follows
#             the approved format: a one-line analogy, then a short Python example,
#             then one common mistake
#   Other user -> 0 episodes retrieved

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import NotRequired, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

store = InMemoryStore(index={"embed": "openai:text-embedding-3-small", "dims": 1536})
model = init_chat_model("openai:gpt-4o-mini", temperature=0)


class State(TypedDict):
    request: str
    answer: NotRequired[str]


def episodes_namespace(user_id: str) -> tuple[str, str, str]:
    return ("assistant", user_id, "episodes")


def format_examples(items: list) -> str:
    if not items:
        return "No approved examples yet."
    blocks = ["Answers this user approved before:"]
    for item in items:
        blocks.append(f"Request: {item.value['request']}\nApproved answer:\n{item.value['answer']}")
    return "\n\n---\n\n".join(blocks)


def answer(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    """Retrieve similar approved episodes, then answer using them as examples."""
    user_id = config["configurable"]["langgraph_user_id"]
    # The application, not the model, decides which past episodes to show.
    matches = store.search(episodes_namespace(user_id), query=state["request"], limit=2)
    print(f"Episodes retrieved: {len(matches)}")

    system = (
        "You explain programming concepts.\n\n"
        f"{format_examples(matches)}\n\n"
        "If approved answers are shown, follow their structure and length closely, "
        "but write new content for the new request."
    )
    reply = model.invoke([{"role": "system", "content": system},
                          {"role": "user", "content": state["request"]}])
    return {"answer": reply.content}


def save_episode(user_id: str, request: str, approved_answer: str) -> None:
    """Save one human-approved exchange. The model never writes episodes itself."""
    store.put(
        episodes_namespace(user_id),
        str(uuid.uuid4()),  # episodes accumulate; each gets its own key
        {"request": request, "answer": approved_answer, "source": "human_approved"},
    )


def build_graph():
    builder = StateGraph(State)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(store=store)  # the Store is injected into `answer`


APPROVED_RECURSION_ANSWER = """Analogy: recursion is a set of Russian dolls; open one to find a smaller one, until the smallest.

```python
def countdown(n):
    if n == 0:          # base case: the smallest doll
        return
    print(n)
    countdown(n - 1)    # a smaller version of the same problem
```

Common mistake: forgetting the base case, which recurses forever."""


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY. Add it to the repository-root .env file.")

    graph = build_graph()
    maya = {"configurable": {"langgraph_user_id": "maya"}}

    print("\n=== BEFORE — no episodes ===")
    before = graph.invoke({"request": "Explain recursion."}, config=maya)
    print(before["answer"][:400], "...")

    print("\n=== THE USER APPROVES AN EDITED ANSWER ===")
    save_episode("maya", "Explain recursion.", APPROVED_RECURSION_ANSWER)

    print("\n=== AFTER — a similar request retrieves the episode ===")
    after = graph.invoke({"request": "Explain dynamic programming."}, config=maya)
    print(after["answer"])

    print("\n=== DIFFERENT USER — Maya's episode is isolated ===")
    graph.invoke({"request": "Explain dynamic programming."},
                 config={"configurable": {"langgraph_user_id": "another-user"}})


if __name__ == "__main__":
    main()
