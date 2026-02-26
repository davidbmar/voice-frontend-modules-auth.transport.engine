"""Tests for engine_starter.llm — StarterLLM."""

import pytest

from engine_starter.interfaces import LLMProvider
from engine_starter.llm import StarterLLM, _resolve_provider


class TestStarterLLM:
    def test_is_llm_provider(self):
        assert issubclass(StarterLLM, LLMProvider)


class TestResolveProvider:
    def test_defaults_to_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _resolve_provider() == "ollama"

    def test_explicit_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "claude")
        assert _resolve_provider() == "claude"

    def test_auto_detect_claude(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _resolve_provider() == "claude"
