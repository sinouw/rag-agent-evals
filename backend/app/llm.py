"""Provider-agnostic chat client.

Groq, Gemini, OpenAI and Ollama all expose an OpenAI-compatible Chat Completions
API, so one client covers them; switching provider is a config change
(`LLM_PROVIDER`, `LLM_MODEL`). `ScriptedLLM` is a deterministic stand-in for tests.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str | None
    key_env: str
    default_model: str


PROVIDERS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": ProviderPreset(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"
    ),
    "openai": ProviderPreset(None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "ollama": ProviderPreset("http://localhost:11434/v1", "OLLAMA_API_KEY", "llama3.1"),
}

# USD per 1M tokens (input, output), from the providers' public price lists.
# Used only to estimate cost in traces and eval reports; override as prices change.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


def estimate_cost(model: str, usage: Usage) -> float | None:
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    return (usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1_000_000


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass
class TextDelta:
    text: str


@dataclass
class TurnEnd:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


StreamItem = TextDelta | TurnEnd


class LLMClient(Protocol):
    provider: str
    model: str

    def stream(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None = None
    ) -> AsyncIterator[StreamItem]: ...

    async def complete(self, messages: Sequence[dict[str, Any]], json_mode: bool = False) -> tuple[str, Usage]: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None,
        temperature: float = 0.0,
        max_retries: int = 6,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=120)

    async def stream(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None = None
    ) -> AsyncIterator[StreamItem]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = list(tools)
        response = await self._client.chat.completions.create(**kwargs)

        text_parts: list[str] = []
        calls: dict[int, ToolCall] = {}
        usage = Usage()
        finish_reason: str | None = None
        async for chunk in response:
            if chunk.usage:
                usage = Usage(chunk.usage.prompt_tokens or 0, chunk.usage.completion_tokens or 0)
            for choice in chunk.choices:
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                if delta.content:
                    text_parts.append(delta.content)
                    yield TextDelta(delta.content)
                for tc in delta.tool_calls or []:
                    # Tool calls arrive in fragments keyed by index; stitch them together.
                    call = calls.setdefault(tc.index, ToolCall(id="", name="", arguments=""))
                    if tc.id:
                        call.id = tc.id
                    if tc.function and tc.function.name:
                        call.name += tc.function.name
                    if tc.function and tc.function.arguments:
                        call.arguments += tc.function.arguments
        tool_calls = [calls[i] for i in sorted(calls)]
        for i, call in enumerate(tool_calls):
            call.id = call.id or f"call_{i}"
        yield TurnEnd("".join(text_parts), tool_calls, usage, finish_reason)

    async def complete(self, messages: Sequence[dict[str, Any]], json_mode: bool = False) -> tuple[str, Usage]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": list(messages), "temperature": 0.0}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)
        usage = Usage()
        if response.usage:
            usage = Usage(response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0)
        return response.choices[0].message.content or "", usage


def create_llm(provider: str, model: str | None = None, api_key: str | None = None, temperature: float = 0.0):
    preset = PROVIDERS.get(provider)
    if preset is None:
        raise ValueError(f"Unknown LLM provider {provider!r}; choose one of {sorted(PROVIDERS)}")
    key = api_key or os.environ.get(preset.key_env) or ("ollama" if provider == "ollama" else None)
    if not key:
        raise RuntimeError(f"No API key for {provider}: set {preset.key_env} (or LLM_API_KEY).")
    return OpenAICompatibleLLM(provider, model or preset.default_model, key, preset.base_url, temperature)


class ScriptedLLM:
    """Replays a fixed list of turns. Each turn is either answer text or a list of tool calls."""

    provider = "scripted"

    def __init__(self, turns: list[str | list[ToolCall]], model: str = "scripted-model") -> None:
        self.model = model
        self._turns = list(turns)
        self.requests: list[list[dict[str, Any]]] = []

    async def stream(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None = None
    ) -> AsyncIterator[StreamItem]:
        self.requests.append(list(messages))
        turn = self._turns.pop(0) if self._turns else "I don't know."
        usage = Usage(input_tokens=sum(len(str(m.get("content") or "")) for m in messages) // 4, output_tokens=10)
        if isinstance(turn, str):
            for word in turn.split(" "):
                yield TextDelta(word + " ")
            yield TurnEnd(turn, [], usage, "stop")
        else:
            yield TurnEnd("", turn, usage, "tool_calls")

    async def complete(self, messages: Sequence[dict[str, Any]], json_mode: bool = False) -> tuple[str, Usage]:
        self.requests.append(list(messages))
        turn = self._turns.pop(0) if self._turns else "{}"
        return str(turn), Usage(10, 10)
