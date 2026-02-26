"""StarterLLM — multi-provider LLM wrapper (Claude, OpenAI, Ollama).

Auto-detects provider from environment: Claude > OpenAI > Ollama.
For production, implement LLMProvider directly with your preferred API.
"""

import asyncio
import functools
import logging
import os

from engine_starter.interfaces import LLMProvider

log = logging.getLogger("engine_starter.llm")

CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
CLAUDE_SONNET = "claude-sonnet-4-6"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"

_anthropic_client = None
_openai_client = None
_httpx_client = None


def _resolve_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider in ("claude", "openai", "ollama"):
        return provider
    if os.getenv("ANTHROPIC_API_KEY", ""):
        return "claude"
    if os.getenv("OPENAI_API_KEY", ""):
        return "openai"
    return "ollama"


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def _get_httpx():
    global _httpx_client
    if _httpx_client is None:
        import httpx
        _httpx_client = httpx.Client(timeout=120.0)
    return _httpx_client


def _generate_claude(system: str, messages: list[dict], model: str = "") -> str:
    client = _get_anthropic()
    active_model = model or CLAUDE_HAIKU
    resp = client.messages.create(
        model=active_model,
        max_tokens=1024 if "sonnet" in active_model else 300,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


def _generate_openai(system: str, messages: list[dict]) -> str:
    client = _get_openai()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_messages = [{"role": "system", "content": system}] + messages
    resp = client.chat.completions.create(model=model, max_tokens=300, messages=openai_messages)
    return resp.choices[0].message.content


def _generate_ollama(system: str, messages: list[dict], model: str = "") -> str:
    client = _get_httpx()
    active_model = model or os.getenv("OLLAMA_MODEL", "phi3:mini")
    ollama_url = os.getenv("OLLAMA_URL", _DEFAULT_OLLAMA_URL)
    ollama_messages = [{"role": "system", "content": system}] + messages
    resp = client.post(
        f"{ollama_url}/api/chat",
        json={"model": active_model, "messages": ollama_messages, "stream": False},
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _generate_sync(system: str, messages: list[dict], provider: str = "", model: str = "") -> str:
    provider = provider or _resolve_provider()
    if provider == "claude":
        return _generate_claude(system, messages, model=model)
    elif provider == "openai":
        return _generate_openai(system, messages)
    else:
        return _generate_ollama(system, messages, model=model)


class StarterLLM(LLMProvider):
    """Reference LLM using Ollama (local), Claude, or OpenAI.

    Auto-detects provider from env vars. Pass provider/model to override.
    """

    def __init__(self, provider: str = "", model: str = "", system_prompt: str = ""):
        self._provider = provider
        self._model = model
        self._system = system_prompt or "You are a helpful voice assistant. Keep responses concise and conversational."

    async def chat(self, messages: list[dict]) -> str:
        loop = asyncio.get_running_loop()
        fn = functools.partial(
            _generate_sync, self._system, messages, self._provider, self._model
        )
        return await loop.run_in_executor(None, fn)
