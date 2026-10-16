"""Tests for ClaudeProvider (examples/llm/providers.py) and the Claude routing for non-Werewolf games."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from altruagent import DecisionContext
from agent.agent import Agent
from examples.llm import providers
from examples.llm.providers import ClaudeProvider, OpenAIProvider, ProviderError, provider_from_env
from examples.llm_agent import LLMAgent


class FakeClient:
    def __init__(self, *answers, fallbacks_ok=True):
        self.answers = list(answers)
        self.requests = []
        self.fallbacks_ok = fallbacks_ok
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._beta))
        self.messages = SimpleNamespace(create=self._create)

    def _beta(self, **request):
        if not self.fallbacks_ok:
            err = Exception("unknown parameter: fallbacks")
            err.status_code = 400
            raise err
        return self._create(**request)

    def _create(self, **request):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, SimpleNamespace):
            return answer
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(answer))])


SCHEMA = {"type": "object", "properties": {"action_id": {"type": "string", "enum": ["a", "b"]},
                                           "n": {"type": "integer", "minimum": 0, "maximum": 3}},
          "required": ["action_id", "n"], "additionalProperties": False}
MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": "state"},
            {"role": "user", "content": "That answer was invalid."}]


def test_answer_is_parsed_and_request_is_shaped_for_claude():
    client = FakeClient({"action_id": "b", "n": 1})
    provider = ClaudeProvider(client, "claude-sonnet-5-5", effort="low")
    assert provider.complete_structured(MESSAGES, "choose_action", SCHEMA, timeout=9.0) == {"action_id": "b", "n": 1}
    req = client.requests[0]
    assert req["system"][0]["text"] == "rules"
    assert [m["role"] for m in req["messages"]] == ["user", "user"]
    assert req["output_config"]["effort"] == "low"
    sent = req["output_config"]["format"]["schema"]
    assert "minimum" not in sent["properties"]["n"] and sent["properties"]["action_id"]["enum"] == ["a", "b"]
    assert req["timeout"] == 9.0


@pytest.mark.parametrize("response, match", [
    (SimpleNamespace(stop_reason="refusal", content=[]), "refused"),
    (SimpleNamespace(stop_reason="max_tokens", content=[]), "out of output tokens"),
    (SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="not json")]), "valid JSON"),
])
def test_bad_replies_raise_provider_error(response, match):
    with pytest.raises(ProviderError, match=match):
        ClaudeProvider(FakeClient(response)).complete_structured(MESSAGES, "x", SCHEMA)


def test_api_errors_become_provider_errors_without_details():
    err = Exception("secret-ish body sk-ant-123")
    err.status_code = 529
    with pytest.raises(ProviderError) as info:
        ClaudeProvider(FakeClient(err)).complete_structured(MESSAGES, "x", SCHEMA)
    assert "sk-ant" not in str(info.value) and "529" in str(info.value)


def test_falls_back_to_plain_requests_when_server_fallbacks_are_unavailable():
    client = FakeClient({"action_id": "a", "n": 0}, fallbacks_ok=False)
    assert ClaudeProvider(client, "claude-sonnet-5-5").complete_structured(MESSAGES, "x", SCHEMA)["action_id"] == "a"


def test_provider_from_env_prefers_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = provider_from_env()
    assert isinstance(provider, ClaudeProvider)
    assert (provider.model, provider.effort) == (providers.DEFAULT_CLAUDE_MODEL, "low") == ("claude-sonnet-5-5", "low")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-haiku-5-5")
    assert provider_from_env().model == "claude-haiku-5-5"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert isinstance(provider_from_env(), OpenAIProvider)


@pytest.mark.parametrize("game", ["pokemon_vgc_doubles_draft", "red_alert"])
def test_router_plays_other_games_on_claude(monkeypatch, game):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    context = DecisionContext(session_id="s", tournament_id=None, game_type=game, agent_id="a", seat_position=0)
    player = Agent()._player(context)
    assert isinstance(player, LLMAgent) and isinstance(player._provider, ClaudeProvider)
