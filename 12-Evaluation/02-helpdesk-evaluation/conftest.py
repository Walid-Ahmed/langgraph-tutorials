"""Keep Chapter 14's local tests independent of tracing settings in .env."""

import pytest


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
