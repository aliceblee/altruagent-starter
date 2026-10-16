"""The model-calling layer for the example LLM agent.

An ``LLMProvider`` has one job: send chat messages and return a JSON object
that matches a given schema. Game logic never talks to a vendor API
directly, so adding another provider (Gemini, a local model) means writing
one class with ``complete_structured`` and choosing it in
``provider_from_env`` — nothing else changes. ``ClaudeProvider`` (the
default when ``ANTHROPIC_API_KEY`` is set) and ``OpenAIProvider`` are here.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

import httpx

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# Pokémon's clocks (15 s per draft pick, 55 s per battle turn) and Red Alert's real time
# want a fast model; Werewolf has its own Claude brain (agent/werewolf/claude_brain.py).
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
DEFAULT_CLAUDE_EFFORT = "low"
CLAUDE_MAX_TOKENS = 16000
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Keywords Claude's structured outputs don't accept; the adapters' own validation
# (InvalidChoice -> one retry) still checks the answer.
_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "maxItems", "uniqueItems", "pattern",
})
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
REQUEST_TIMEOUT_SECONDS = 60.0


class ProviderError(RuntimeError):
    """The model request failed or its reply wasn't a JSON object."""


class LLMProvider(Protocol):
    model: str

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float | None = None
    ) -> dict:
        """Return the model's answer as a dict matching ``schema``, or raise
        ``ProviderError``. ``timeout`` (seconds), when given, bounds this one
        request: a game with a clock (Pokémon) passes what's left of the
        decision's time. A provider without the parameter still works; the
        agent then only limits how many requests it makes.
        """
        ...


class OpenAIProvider:
    """OpenAI Chat Completions with strict structured output, over the
    starter's existing ``httpx`` dependency (no SDK).

    The API key only ever goes into the Authorization header — never into
    ``repr``, logs, or error messages (provider error bodies can echo parts
    of the key, so only the HTTP status is reported).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model!r})"

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float | None = None
    ) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        # Only a game with a clock passes a timeout; otherwise the client's own applies.
        extra = {"timeout": timeout} if timeout is not None else {}
        try:
            response = self._http.post("/chat/completions", json=body, **extra)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI request failed ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise ProviderError(f"OpenAI request failed (HTTP {response.status_code})")
        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("OpenAI response had no message") from None
        if message.get("refusal"):
            raise ProviderError("the model refused to answer")
        try:
            data = json.loads(message.get("content") or "")
        except ValueError:
            raise ProviderError("the model's reply was not valid JSON") from None
        if not isinstance(data, dict):
            raise ProviderError("the model's reply was not a JSON object")
        return data


class ClaudeProvider:
    """Claude through the official Anthropic SDK, with structured outputs.

    System messages become Claude's system prompt; the rest keep their roles.
    A declined request is retried server-side on another model when the model
    supports it (``fallbacks: "default"``). The SDK reads ``ANTHROPIC_API_KEY``
    itself; the key never appears in ``repr``, logs, or error messages.
    """

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_CLAUDE_MODEL,
        *,
        effort: str = DEFAULT_CLAUDE_EFFORT,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=1, timeout=timeout)
        self._client = client
        self.model = model
        self.effort = effort
        self._use_fallbacks = model.startswith(("claude-opus-5", "claude-sonnet-5"))

    def __repr__(self) -> str:
        return f"ClaudeProvider(model={self.model!r}, effort={self.effort!r})"

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float | None = None
    ) -> dict:
        system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system")
        turns = [{"role": m["role"], "content": str(m.get("content") or "")}
                 for m in messages if m.get("role") in ("user", "assistant")]
        request: dict = {
            "model": self.model,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": turns,
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema", "schema": _claude_schema(schema)}},
        }
        if system:
            request["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if timeout is not None:
            request["timeout"] = timeout
        try:
            response = self._create(request)
        except Exception as exc:  # noqa: BLE001 - any SDK/network error becomes a ProviderError
            status = getattr(exc, "status_code", None)
            raise ProviderError(f"Claude request failed ({type(exc).__name__}"
                                f"{f', HTTP {status}' if status else ''})") from None
        if response.stop_reason == "refusal":
            raise ProviderError("the model refused to answer")
        if response.stop_reason == "max_tokens":
            raise ProviderError("the model ran out of output tokens")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        try:
            data = json.loads(text)
        except ValueError:
            raise ProviderError("the model's reply was not valid JSON") from None
        if not isinstance(data, dict):
            raise ProviderError("the model's reply was not a JSON object")
        return data

    def _create(self, request: dict) -> Any:
        if self._use_fallbacks:
            try:
                return self._client.beta.messages.create(betas=[CLAUDE_FALLBACK_BETA], fallbacks="default", **request)
            except Exception as exc:  # noqa: BLE001
                if getattr(exc, "status_code", None) != 400 or "fallback" not in str(exc).lower():
                    raise
                self._use_fallbacks = False  # not available for this model/account: plain requests from now on
        return self._client.messages.create(**request)


def _claude_schema(schema: Any) -> Any:
    """``schema`` without the keywords Claude's structured outputs reject (and
    ``minItems`` above 1)."""
    if isinstance(schema, list):
        return [_claude_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS or (key == "minItems" and isinstance(value, int) and value > 1):
            continue
        out[key] = value if key in ("enum", "const", "required") else _claude_schema(value)
    return out


def provider_from_env() -> LLMProvider:
    """Claude when ``ANTHROPIC_API_KEY`` is set (``CLAUDE_MODEL``, default
    ``claude-sonnet-5-5``; ``CLAUDE_EFFORT``, default ``low``), else OpenAI
    (``OPENAI_API_KEY``, ``OPENAI_MODEL``, ``OPENAI_BASE_URL``).
    ``LLM_PROVIDER=openai`` picks OpenAI even with both keys set."""
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("LLM_PROVIDER", "").lower() != "openai":
        return ClaudeProvider(
            model=os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
            effort=os.environ.get("CLAUDE_EFFORT") or DEFAULT_CLAUDE_EFFORT,
        )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set — add one to your environment "
                           "or the starter's .env.")
    return OpenAIProvider(
        api_key,
        os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        base_url=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
    )
