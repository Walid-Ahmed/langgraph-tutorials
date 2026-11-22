"""Settings, and the chat models built from them.

Everything configurable lives here, so there is exactly one file that knows
which environment variables exist. Read them from a ``.env`` file in the
project root (copy ``.env.example`` to get started).

    settings = load_settings()
    model = get_chat_model(settings)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

# ltm_chat/config.py -> ltm_chat/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MissingAPIKeyError(RuntimeError):
    """Raised when a real model is asked for but no API key is configured."""


@dataclass(frozen=True)
class Settings:
    """All configuration for one run of the app."""

    api_key: str | None
    base_url: str | None  # optional: any OpenAI-compatible endpoint
    chat_model: str  # the model that talks to the user
    memory_model: str  # the model that rewrites the user's profile
    temperature: float
    backend: str  # "sqlite" (survives restarts) or "memory" (lost on exit)
    data_dir: Path  # where the two SQLite files live

    @property
    def store_path(self) -> Path:
        """SQLite file for long-term memory (the store)."""
        return self.data_dir / "long_term_memory.sqlite"

    @property
    def checkpoint_path(self) -> Path:
        """SQLite file for short-term memory (the checkpointer)."""
        return self.data_dir / "checkpoints.sqlite"


def load_settings() -> Settings:
    """Read the .env file and the environment into one Settings object."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    backend = os.getenv("LTM_BACKEND", "sqlite").strip().lower()
    if backend not in ("sqlite", "memory"):
        raise ValueError(f"LTM_BACKEND must be 'sqlite' or 'memory', got {backend!r}")

    chat_model = os.getenv("LTM_CHAT_MODEL", "gpt-4o-mini")
    temperature = os.getenv("LTM_CHAT_TEMPERATURE") or "0.7"

    return Settings(
        api_key=os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        chat_model=chat_model,
        # The memory model defaults to the chat model.
        memory_model=os.getenv("LTM_MEMORY_MODEL", chat_model),
        temperature=float(temperature),
        backend=backend,
        data_dir=Path(os.getenv("LTM_DATA_DIR", PROJECT_ROOT / "data")).expanduser(),
    )


def _build_model(model: str, temperature: float, settings: Settings) -> BaseChatModel:
    if not settings.api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # Imported here so the rest of the project can be used without langchain-openai.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )


def get_chat_model(settings: Settings) -> BaseChatModel:
    """The model that answers the user (some creativity allowed)."""
    return _build_model(settings.chat_model, settings.temperature, settings)


def get_memory_model(settings: Settings) -> BaseChatModel:
    """The model that rewrites the profile (temperature 0 = predictable)."""
    return _build_model(settings.memory_model, 0.0, settings)
