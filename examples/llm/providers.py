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
import re
from typing import Any, Protocol

import httpx

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# Pokémon's clocks (15 s per draft pick, 55 s per battle turn) and Red Alert's real time
# want a fast model; Werewolf has its own Claude brain (agent/werewolf/claude_brain.py).
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
DEFAULT_CLAUDE_EFFORT = "low"
# Thinking off by default: with it on, Sonnet 5.5 took 11-14 s per Pokémon draft pick in
# testing, past our 10 s budget (the server allows 15), so most picks fell back.
DEFAULT_CLAUDE_THINKING = "off"
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
        thinking: str = DEFAULT_CLAUDE_THINKING,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if client is None:
            import anthropic

            # No SDK retries: the LLM agent retries itself, and a dead connection should fail
            # fast enough to leave time for the fallback (test games lost ~4.5 s per failed call).
            client = anthropic.Anthropic(max_retries=0, timeout=timeout)
        self._client = client
        self.model = model
        self.effort = effort
        self._thinking = _thinking_off(model) if thinking == "off" else None
        # Server-side fallbacks re-run the same request on another model, which would reject a
        # model-specific "thinking off" setting; Pokémon and Red Alert rarely trigger refusals.
        self._use_fallbacks = self._thinking is None and model.startswith(("claude-opus-5", "claude-sonnet-5"))

    def __repr__(self) -> str:
        return f"ClaudeProvider(model={self.model!r}, effort={self.effort!r}, thinking={'off' if self._thinking else 'on'})"

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
        if self._thinking is not None:
            request["thinking"] = self._thinking
        if system:
            request["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if timeout is not None:
            request["timeout"] = timeout
        try:
            response = self._create(request)
        except Exception as exc:  # noqa: BLE001 - any SDK/network error becomes a ProviderError
            status = getattr(exc, "status_code", None)
            detail = ""
            if isinstance(status, int) and 400 <= status < 500 and status not in (401, 403):
                # The API's own message ("thinking.type ... is not supported") says what to fix.
                detail = ": " + re.sub(r"sk-ant-\S+", "[key]", str(getattr(exc, "message", "") or ""))[:300]
            if status is None:
                detail = _root_cause(exc)  # e.g. ": ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] ..."
            raise ProviderError(f"Claude request failed ({type(exc).__name__}"
                                f"{f', HTTP {status}' if status else ''}){detail}") from None
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


def _root_cause(exc: BaseException) -> str:
    """The innermost cause of a connection error, as ": Type: message" (no secrets: these
    come from the network stack, not from the request)."""
    seen, inner = set(), exc
    while (inner.__cause__ or inner.__context__) is not None and id(inner) not in seen:
        seen.add(id(inner))
        inner = inner.__cause__ or inner.__context__
    if inner is exc:
        return ""
    return ": " + re.sub(r"sk-ant-\S+", "[key]", f"{type(inner).__name__}: {inner}")[:300]


class FailoverProvider:
    """Claude first; OpenAI for any call where Claude can't be reached, and for the rest of
    the match after ``SWITCH_AFTER`` connection failures in a row. Invalid answers are not
    failed over (the agent's own retry handles those)."""

    SWITCH_AFTER = 2

    def __init__(self, primary: Any, backup: Any, log: Any = print) -> None:
        self.primary, self.backup, self._log = primary, backup, log
        self._failures = 0
        self._switched = False

    @property
    def model(self) -> str:
        return self.backup.model if self._switched else self.primary.model

    def __repr__(self) -> str:
        return f"FailoverProvider({self.primary!r} -> {self.backup!r})"

    def complete_structured(
        self, messages: list[dict], schema_name: str, schema: dict, *, timeout: float | None = None
    ) -> dict:
        if not self._switched:
            try:
                answer = self.primary.complete_structured(messages, schema_name, schema, timeout=timeout)
                self._failures = 0
                return answer
            except ProviderError as exc:
                if "APIConnectionError" not in str(exc) and "APITimeoutError" not in str(exc):
                    raise
                self._failures += 1
                if self._failures >= self.SWITCH_AFTER:
                    self._switched = True
                    self._log(f"[llm] Claude unreachable {self._failures} times in a row ({exc}); "
                              f"using {self.backup.model} for the rest of this match")
                else:
                    self._log(f"[llm] Claude unreachable ({exc}); this call goes to {self.backup.model}")
        return self.backup.complete_structured(messages, schema_name, schema, timeout=timeout)


def _thinking_off(model: str) -> dict | None:
    """The request's "no extended thinking" setting for ``model``, or None where thinking
    can't be turned off (Claude Opus 5.5: lower ``effort`` instead)."""
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}
    if model.startswith("claude-haiku-5"):
        return {"type": "disabled"}
    return None


def _claude_schema(schema: Any) -> Any:
    """``schema`` without the keywords Claude's structured outputs reject (and
    ``minItems`` above 1), with ``"type": [a, b]`` written as ``anyOf``."""
    if isinstance(schema, list):
        return [_claude_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if isinstance(schema.get("type"), list):
        rest = {k: v for k, v in schema.items() if k != "type"}
        return _claude_schema({"anyOf": [{**rest, "type": t} if t != "null" else {"type": "null"} for t in schema["type"]]})
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
    ``LLM_PROVIDER=openai`` picks OpenAI even with both keys set. With both keys,
    calls fail over to OpenAI when Claude can't be reached (``FailoverProvider``)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("LLM_PROVIDER", "").lower() != "openai":
        claude = ClaudeProvider(
            model=os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
            effort=os.environ.get("CLAUDE_EFFORT") or DEFAULT_CLAUDE_EFFORT,
            thinking=(os.environ.get("CLAUDE_THINKING") or DEFAULT_CLAUDE_THINKING).lower(),
        )
        if not api_key:
            return claude
        openai = OpenAIProvider(api_key, os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
                                base_url=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL)
        return FailoverProvider(claude, openai)
    if not api_key:
        raise RuntimeError("Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set — add one to your environment "
                           "or the starter's .env.")
    return OpenAIProvider(
        api_key,
        os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        base_url=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
    )
