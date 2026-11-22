"""Streamlit UI for the long-term memory chatbot.

    streamlit run app.py

Layout:
    sidebar       pick the user and the conversation, run the demo, reset
    left column   the chat itself
    right column  the inspector, which shows what each kind of memory holds

This file only draws things. Everything it knows comes from
MemoryChatService, in ltm_chat/service.py.
"""

from __future__ import annotations

import json

import streamlit as st

# Must be the first Streamlit call, so it comes before our own imports.
st.set_page_config(page_title="LangGraph Long-Term Memory", page_icon="🧠", layout="wide")

from ltm_chat import MemoryChatService, TurnResult, load_settings, new_thread_id  # noqa: E402
from ltm_chat.config import MissingAPIKeyError  # noqa: E402
from ltm_chat.memory import PROFILE_KEY, profile_ns  # noqa: E402

AVATARS = {"human": "🧑‍💻", "ai": "🤖"}

DEMO_1 = (
    "Hey, how are you? I am Evgeny, I am a software engineer and I need your help "
    "with my project. This is a TaskManager Java Spring Boot application."
)
DEMO_2 = "How can I set up security configurations?"


# --------------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Opening the memory backends…")
def get_service() -> MemoryChatService:
    """Built once per server process, not once per rerun.

    This matters: with the in-memory backend, rebuilding the service would
    mean a brand new, empty store every time you click something.
    """
    return MemoryChatService.from_settings(load_settings())


def init_state() -> None:
    """Session state holds UI state only. Real memory lives in the store."""
    st.session_state.setdefault("user_id", "1")
    st.session_state.setdefault("thread_id", new_thread_id())
    st.session_state.setdefault("last_turn", None)
    st.session_state.setdefault("pending_prompt", None)


def start_new_thread() -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.last_turn = None


# ------------------------------------------------------------------- sidebar
def sidebar(svc: MemoryChatService) -> None:
    with st.sidebar:
        st.title("🧠 LTM Chat")
        st.caption(f"Backend: **{svc.backends.description}**")

        st.subheader("👤 User", help="user_id chooses which long-term memory to use.")
        typed = st.text_input("User ID", value=st.session_state.user_id).strip()
        if typed and typed != st.session_state.user_id:
            st.session_state.user_id = typed
            start_new_thread()
            st.rerun()

        others = [u for u in svc.list_users() if u != st.session_state.user_id]
        if others:
            st.caption("Known users")
            for uid in others[:8]:
                if st.button(uid, key=f"user_{uid}", use_container_width=True):
                    st.session_state.user_id = uid
                    start_new_thread()
                    st.rerun()

        st.divider()
        st.subheader("🧵 Conversation", help="thread_id chooses the short-term memory.")
        if st.button("➕ New conversation", use_container_width=True, type="primary"):
            start_new_thread()
            st.rerun()

        threads = svc.list_threads(st.session_state.user_id)
        current = st.session_state.thread_id
        if not any(t.thread_id == current for t in threads):
            st.caption(f"`{current}` — new, nothing sent yet")
        for t in threads:
            is_current = t.thread_id == current
            if st.button(
                ("▶ " if is_current else "") + t.title,
                key=f"thread_{t.thread_id}",
                use_container_width=True,
                disabled=is_current,
                help=f"thread_id={t.thread_id}",
            ):
                st.session_state.thread_id = t.thread_id
                st.session_state.last_turn = None
                st.rerun()

        st.divider()
        with st.expander("🎬 Try the demo", expanded=True):
            st.caption(
                "Introduce yourself in one conversation, then start a new one "
                "and watch the bot still know who you are."
            )
            if st.button("1 · Introduce yourself", use_container_width=True):
                start_new_thread()
                st.session_state.pending_prompt = DEMO_1
                st.rerun()
            if st.button("2 · New conversation, ask something", use_container_width=True):
                start_new_thread()
                st.session_state.pending_prompt = DEMO_2
                st.rerun()

        with st.expander("🧹 Reset"):
            st.caption("Deletes this user's profile and its history. Conversations stay.")
            if st.button("Forget this user", use_container_width=True):
                svc.forget_user(st.session_state.user_id)
                st.session_state.last_turn = None
                st.rerun()


# ---------------------------------------------------------------------- chat
def run_turn(svc: MemoryChatService, text: str) -> None:
    with st.chat_message("human", avatar=AVATARS["human"]):
        st.markdown(text)

    with st.chat_message("ai", avatar=AVATARS["ai"]):
        box = st.empty()
        shown: list[str] = []

        def on_token(token: str) -> None:
            shown.append(token)
            box.markdown("".join(shown))

        turn = svc.chat(
            st.session_state.user_id,
            st.session_state.thread_id,
            text,
            on_token=on_token,
        )
        box.markdown(turn.reply)

    st.session_state.last_turn = turn
    # Rerun so the sidebar and the inspector show what this turn did.
    st.rerun()


def chat_pane(svc: MemoryChatService) -> None:
    thread_id = st.session_state.thread_id
    st.markdown(f"#### 💬 Chat `user_id={st.session_state.user_id}` `thread_id={thread_id}`")

    box = st.container(height=560, border=True)
    with box:
        messages = svc.get_messages(thread_id)
        for msg in messages:
            if msg.type in AVATARS:
                with st.chat_message(msg.type, avatar=AVATARS[msg.type]):
                    st.markdown(msg.content)
        if not messages:
            st.caption(
                "Empty conversation. Say who you are and what you work on, then "
                "start a new conversation and see whether the bot remembers."
            )

    prompt = st.chat_input("Message the assistant…")
    if st.session_state.pending_prompt:  # queued by a demo button
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    if prompt:
        with box:
            run_turn(svc, prompt)


# ----------------------------------------------------------------- inspector
def _changed_lines(turn: TurnResult) -> None:
    def lines(text: str | None) -> list[str]:
        return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    if not turn.memory_updated:
        st.info("Last turn learned nothing new (the model answered NO_UPDATE).", icon="💤")
        return

    st.success("Last turn updated long-term memory", icon="🧠")
    before, after = lines(turn.memory_before), lines(turn.memory_after)
    for ln in [x for x in after if x not in before]:
        st.markdown(f"🟢 {ln.lstrip('-• ')}")
    for ln in [x for x in before if x not in after]:
        st.markdown(f"🔴 ~~{ln.lstrip('-• ')}~~")


def inspector(svc: MemoryChatService) -> None:
    user_id = st.session_state.user_id
    thread_id = st.session_state.thread_id
    st.markdown("#### 🔍 Memory inspector")
    long_tab, short_tab, history_tab, store_tab, graph_tab = st.tabs(
        ["🧠 Long-term", "💬 Short-term", "History", "Store", "Graph"]
    )

    with long_tab:
        st.caption(
            f"Namespace `{profile_ns(user_id)}`, key `{PROFILE_KEY}` — "
            "shared by **every conversation** of this user."
        )
        if st.session_state.last_turn:
            _changed_lines(st.session_state.last_turn)
        profile = svc.get_profile(user_id)
        if profile is None:
            st.warning("Nothing known about this user yet.", icon="🫙")
        else:
            with st.container(border=True):
                st.markdown(profile.text)
            st.caption(f"Updated {profile.saved_at:%Y-%m-%d %H:%M:%S} UTC")

    with short_tab:
        messages = svc.get_messages(thread_id)
        st.caption(
            f"Checkpoints for `thread_id={thread_id}` — **only this conversation** "
            "sees them. A new conversation starts empty; long-term memory does not."
        )
        c1, c2 = st.columns(2)
        c1.metric("Messages", len(messages))
        c2.metric("Checkpoints", svc.checkpoint_count(thread_id))
        for i, msg in enumerate(messages):
            preview = " ".join(str(msg.content).split())[:110]
            st.markdown(f"`{i:02d}` **{msg.type}** — {preview}")

    with history_tab:
        versions = svc.get_history(user_id)
        if not versions:
            st.caption("No versions yet.")
        else:
            st.caption(f"{len(versions)} version(s), newest first.")
            for n, v in reversed(list(enumerate(versions, start=1))):
                label = f"v{n} · {v.saved_at:%H:%M:%S} · thread `{v.thread_id}`"
                with st.expander(label, expanded=n == len(versions)):
                    st.markdown(v.text)

    with store_tab:
        st.caption("Everything in the store, grouped by namespace.")
        dump = svc.store_dump()
        if not dump:
            st.caption("The store is empty.")
        for ns, items in sorted(dump.items()):
            with st.expander(f"{ns} — {len(items)} item(s)"):
                st.code(json.dumps(items, indent=2, default=str), language="json")

    with graph_tab:
        st.caption("`chat` answers the user · `update_memory` rewrites the profile")
        mermaid = svc.graph_mermaid()
        # Drawn in the browser by mermaid.js. The text is our own compiled
        # graph, never anything the user typed.
        html = (
            f'<div class="mermaid" style="display:flex;justify-content:center">{mermaid}</div>'
            '<script type="module">'
            'import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";'
            'mermaid.initialize({ startOnLoad: true, theme: "neutral" });'
            "</script>"
        )
        if hasattr(st, "iframe"):  # Streamlit >= 1.52
            st.iframe(html, height=360)
        else:
            import streamlit.components.v1 as components

            components.html(html, height=360)
        with st.expander("Mermaid source"):
            st.code(mermaid, language="text")


# ---------------------------------------------------------------------- main
def main() -> None:
    init_state()
    try:
        svc = get_service()
    except MissingAPIKeyError as exc:
        st.error(f"**Setup needed:** {exc}")
        st.code("copy .env.example .env\n# then put your key in OPENAI_API_KEY", language="bash")
        st.stop()

    sidebar(svc)
    st.markdown(
        "**Short-term memory** is the checkpointer, scoped to one `thread_id`. "
        "**Long-term memory** is the store, scoped to a `user_id` and shared by "
        "all of their conversations."
    )
    left, right = st.columns([3, 2], gap="large")
    with right:
        inspector(svc)
    with left:
        chat_pane(svc)  # last, because sending a message triggers a rerun


main()
