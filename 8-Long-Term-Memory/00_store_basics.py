# Demonstrates LangGraph Store operations without an LLM or graph.
#
# A Store organizes long-term memory as:
#   namespace + key -> value
#
# Run from the repository root:
#   python "8-Long-Term-Memory/00_store_basics.py"

from langgraph.store.memory import InMemoryStore


def main() -> None:
    store = InMemoryStore()

    # A namespace behaves like a folder path. Including user_id keeps one
    # user's memories isolated from every other user's memories.
    user_id = "walid"
    namespace = (user_id, "memories")

    # A key identifies one entry inside that namespace, and the value is the
    # dictionary containing the information we want to remember.
    store.put(
        namespace,
        "profile",
        {
            "name": "Walid",
            "role": "software engineer",
            "preference": "concise explanations",
        },
    )
    store.put(
        namespace,
        "learning_goal",
        {"topic": "LangGraph long-term memory"},
    )

    # get(namespace, key) fetches one exact memory.
    profile = store.get(namespace, "profile")
    print("One memory:")
    print(profile.value if profile else "Profile not found")

    if profile:
        print("\nStored Item metadata:")
        print(f"- namespace:  {profile.namespace}")
        print(f"- key:        {profile.key}")
        print(f"- created_at: {profile.created_at}")
        print(f"- updated_at: {profile.updated_at}")

    # search(namespace) lists all memories in the namespace.
    print("\nAll memories for this user:")
    for item in store.search(namespace):
        print(f"- {item.key}: {item.value}")

    # Reusing the same namespace + key updates that entry.
    store.put(
        namespace,
        "profile",
        {
            "name": "Walid",
            "role": "software engineer",
            "preference": "step-by-step explanations",
        },
    )
    updated_profile = store.get(namespace, "profile")
    print("\nUpdated profile:")
    print(updated_profile.value if updated_profile else "Profile not found")


if __name__ == "__main__":
    main()
