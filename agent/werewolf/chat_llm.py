"""Optional LLM polish for chat messages.

The rule-based planner decides *what* to say (claims, accusations, vote
target). The model only rewrites that into something more natural and
persuasive. It is never told our real role, so it can't leak it, and its
rewrite is rejected (falling back to the planned text) if it drops the vote
target, mentions a player the plan didn't, or adds a seer claim the plan
didn't make.

Enabled when ``OPENAI_API_KEY`` is set (same provider config as
``examples/llm_agent.py``: ``OPENAI_MODEL``, ``OPENAI_BASE_URL``). Disable
with ``WEREWOLF_LLM_CHAT=0``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

from .parsing import _PLAYER_RE, _SEER_CLAIM, plain_quotes

TIMEOUT_SECONDS = 12.0
MAX_WORDS = 45

_SYSTEM = (
    "You write one chat message for a player in a 7-player Werewolf game (2 wolves, 1 seer, 4 villagers). "
    "Rewrite the draft so it sounds like a confident, persuasive human player. Keep every factual claim, "
    "every player name exactly as written (e.g. 'Player3'), and the stated vote. Do not add new claims, "
    "new player names, or new roles. At most 40 words. Plain text, no quotes, no emojis."
)

_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}


def _players(text: str) -> set[int]:
    return {int(m.group(1)) for m in _PLAYER_RE.finditer(text)}


def make_polisher(provider: Any, log: Callable[[str], None] = lambda _m: None) -> Callable[[str, dict], str]:
    def polish(draft: str, facts: dict) -> str:
        system = _SYSTEM
        if facts.get("voice"):
            system += f" Write in a {facts['voice']} voice."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Draft: {draft}"},
        ]
        try:
            answer = provider.complete_structured(messages, "werewolf_chat", _SCHEMA, timeout=TIMEOUT_SECONDS)
        except TypeError:
            answer = provider.complete_structured(messages, "werewolf_chat", _SCHEMA)
        text = " ".join(str(answer.get("message") or "").split()[:MAX_WORDS])
        if not text:
            return draft
        if not _players(text) <= _players(draft):
            log("[werewolf] polish rejected: new player mentioned")
            return draft
        vote = facts.get("vote")
        if isinstance(vote, int) and not re.search(rf"\bplayer\s*{vote}\b", text, re.I):
            log("[werewolf] polish rejected: vote target dropped")
            return draft
        if _SEER_CLAIM.search(plain_quotes(text)) and not _SEER_CLAIM.search(plain_quotes(draft)):
            log("[werewolf] polish rejected: invented a seer claim")
            return draft
        if facts.get("intent") == "seer_claim" and not _SEER_CLAIM.search(plain_quotes(text)):
            log("[werewolf] polish rejected: dropped the seer claim")
            return draft
        return text

    return polish


def polisher_from_env(log: Callable[[str], None] = lambda _m: None) -> Callable[[str, dict], str] | None:
    if os.environ.get("WEREWOLF_LLM_CHAT", "1") == "0" or not os.environ.get("OPENAI_API_KEY"):
        return None
    from examples.llm.providers import provider_from_env

    return make_polisher(provider_from_env(), log)
