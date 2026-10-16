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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = provider_from_env()
    assert isinstance(provider, ClaudeProvider)
    assert (provider.model, provider.effort) == (providers.DEFAULT_CLAUDE_MODEL, "low") == ("claude-sonnet-5-5", "low")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-haiku-5-5")
    assert provider_from_env().model == "claude-haiku-5-5"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert isinstance(provider_from_env(), OpenAIProvider)


@pytest.mark.parametrize("game", ["pokemon_vgc_doubles_draft", "red_alert"])
def test_router_plays_other_games_on_claude(monkeypatch, game):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    context = DecisionContext(session_id="s", tournament_id=None, game_type=game, agent_id="a", seat_position=0)
    player = Agent()._player(context)
    assert isinstance(player, LLMAgent) and isinstance(player._provider, ClaudeProvider)


def test_router_logs_our_moves_and_outcome_in_other_games(tmp_path, monkeypatch):
    from altruagent.models import GameState
    from agent.decision_log import DecisionLog

    class Delegate:
        def choose_action(self, state, context):
            return {"type": "doubles_turn"}

    monkeypatch.setenv("ALTRUAGENT_DECISION_LOG_DIR", str(tmp_path))
    agent = Agent(delegate=Delegate(), decision_log=DecisionLog())
    context = DecisionContext(session_id="pk", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                              agent_id="a", seat_position=0)
    state = GameState.from_mcp_state({"session_id": "pk", "game_type": "pokemon_vgc_doubles_draft", "phase": "moving",
                                      "your_position": 0, "players": []})
    assert agent.choose_action(state, context) == {"type": "doubles_turn"}
    agent.on_game_end(None, context, "won")
    lines = [json.loads(line) for line in (tmp_path / "pk" / "seat0.jsonl").read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["action", "game_end"]
    assert "doubles_turn" in lines[0]["decision"] and lines[1]["outcome"] == "won"


@pytest.mark.parametrize("model, thinking", [
    ("claude-sonnet-5-5", {"type": "between_tools"}),
    ("claude-haiku-5-5", {"type": "disabled"}),
    ("claude-opus-5-5", None),  # can't be turned off: effort controls it
])
def test_thinking_is_off_by_default_and_server_fallbacks_are_skipped(model, thinking):
    client = FakeClient({"action_id": "a", "n": 0}, fallbacks_ok=False)  # a beta call would raise
    ClaudeProvider(client, model).complete_structured(MESSAGES, "x", SCHEMA)
    assert client.requests[0].get("thinking") == thinking


def test_thinking_on_keeps_adaptive_thinking():
    client = FakeClient({"action_id": "a", "n": 0})
    ClaudeProvider(client, "claude-sonnet-5-5", thinking="on").complete_structured(MESSAGES, "x", SCHEMA)
    assert "thinking" not in client.requests[0]


def test_type_lists_become_any_of():
    client = FakeClient({"t": None})
    schema = {"type": "object", "properties": {"t": {"type": ["integer", "null"]}}, "required": ["t"],
              "additionalProperties": False}
    ClaudeProvider(client).complete_structured(MESSAGES, "x", schema)
    sent = client.requests[0]["output_config"]["format"]["schema"]["properties"]["t"]
    assert sent == {"anyOf": [{"type": "integer"}, {"type": "null"}]}


def test_bad_request_errors_say_why():
    err = Exception("bad")
    err.status_code = 400
    err.message = "thinking.type.between_tools is not supported for this model"
    with pytest.raises(ProviderError, match="not supported for this model"):
        ClaudeProvider(FakeClient(err)).complete_structured(MESSAGES, "x", SCHEMA)


class _Stub:
    def __init__(self, model, *results):
        self.model, self.results, self.calls = model, list(results), 0

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_failover_uses_openai_when_claude_is_unreachable_then_sticks_to_it():
    from examples.llm.providers import FailoverProvider

    down = ProviderError("Claude request failed (APIConnectionError): ConnectError: refused")
    claude = _Stub("claude-sonnet-5-5", down, down)
    openai = _Stub("gpt", {"a": 1}, {"a": 2}, {"a": 3})
    provider = FailoverProvider(claude, openai, log=lambda _m: None)
    assert [provider.complete_structured([], "x", {}) for _ in range(3)] == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert claude.calls == 2 and provider.model == "gpt"  # stopped trying Claude after 2 failures


def test_failover_does_not_hide_bad_answers():
    from examples.llm.providers import FailoverProvider

    provider = FailoverProvider(_Stub("c", ProviderError("the model's reply was not valid JSON")), _Stub("gpt"))
    with pytest.raises(ProviderError, match="not valid JSON"):
        provider.complete_structured([], "x", {})


def test_connection_errors_name_their_root_cause():
    try:
        try:
            raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        except OSError as inner:
            raise RuntimeError("Connection error.") from inner
    except RuntimeError as exc:
        err = exc
    with pytest.raises(ProviderError, match="CERTIFICATE_VERIFY_FAILED"):
        ClaudeProvider(FakeClient(err)).complete_structured(MESSAGES, "x", SCHEMA)


def test_with_both_keys_other_games_fail_over_to_openai(monkeypatch):
    from examples.llm.providers import FailoverProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = provider_from_env()
    assert isinstance(provider, FailoverProvider) and isinstance(provider.primary, ClaudeProvider)
