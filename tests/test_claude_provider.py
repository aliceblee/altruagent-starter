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
def test_thinking_is_off_by_default_for_quick_decisions_and_server_fallbacks_are_skipped(model, thinking):
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


def test_too_large_grammar_falls_back_to_plain_json_and_remembers():
    err = Exception("Error code: 400 - The compiled grammar is too large, which would cause performance issues")
    err.status_code = 400
    fenced = SimpleNamespace(stop_reason="end_turn",
                             content=[SimpleNamespace(type="text", text='```json\n{"orders": []}\n```')])
    client = FakeClient(err, fenced, {"orders": [1]})
    provider = ClaudeProvider(client, "claude-sonnet-5-5")
    assert provider.complete_structured(MESSAGES, "red_alert_orders", SCHEMA) == {"orders": []}
    assert provider.complete_structured(MESSAGES, "red_alert_orders", SCHEMA) == {"orders": [1]}
    retry, later = client.requests[1], client.requests[2]
    for req in (retry, later):
        assert "format" not in req["output_config"] and "JSON schema" in req["messages"][-1]["content"]


def test_auto_thinking_is_on_for_team_preview_and_battle_turns():
    client = FakeClient({"action_id": "a", "n": 0}, {"action_id": "a", "n": 0}, {"action_id": "a", "n": 0})
    provider = ClaudeProvider(client, "claude-sonnet-5-5")
    for name in ("select_lineup", "doubles_turn", "choose_action"):
        provider.complete_structured(MESSAGES, name, SCHEMA)
    assert [r.get("thinking") for r in client.requests] == [None, None, {"type": "between_tools"}]


def test_out_of_credit_fails_over_to_openai():
    from examples.llm.providers import FailoverProvider

    broke = ProviderError("Claude request failed (BadRequestError, HTTP 400): Your credit balance is too low to access "
                          "the Anthropic API.")
    claude, openai = _Stub("c", broke, broke), _Stub("gpt", {"a": 1}, {"a": 2}, {"a": 3})
    provider = FailoverProvider(claude, openai, log=lambda _m: None)
    assert [provider.complete_structured([], "x", {}) for _ in range(3)] == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert claude.calls == 2


def test_battle_turns_get_medium_effort_draft_picks_stay_low():
    client = FakeClient({"action_id": "a", "n": 0}, {"action_id": "a", "n": 0})
    provider = ClaudeProvider(client, "claude-sonnet-5-5", effort="low")
    provider.complete_structured(MESSAGES, "doubles_turn", SCHEMA)
    provider.complete_structured(MESSAGES, "choose_action", SCHEMA)
    assert [r["output_config"]["effort"] for r in client.requests] == ["medium", "low"]


def test_team_preview_and_battle_turns_use_the_deep_model_until_it_times_out():
    class APITimeoutError(Exception):
        pass

    now = [0.0]
    ok = {"action_id": "a", "n": 0}
    client = FakeClient(ok, ok, APITimeoutError("slow"), ok, ok)
    provider = ClaudeProvider(client, "claude-sonnet-5-5", clock=lambda: now[0])
    provider.complete_structured(MESSAGES, "doubles_turn", SCHEMA)
    provider.complete_structured(MESSAGES, "choose_action", SCHEMA)  # draft pick: fast model
    with pytest.raises(ProviderError):
        provider.complete_structured(MESSAGES, "doubles_turn", SCHEMA)
    provider.complete_structured(MESSAGES, "doubles_turn", SCHEMA)  # cooling down: base model
    now[0] += 200
    provider.complete_structured(MESSAGES, "doubles_turn", SCHEMA)
    assert [r["model"] for r in client.requests] == [
        "claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
    assert "thinking" not in client.requests[0]  # Opus thinks; no Sonnet-only setting sent


def test_deep_model_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_DEEP_MODEL", "")
    assert provider_from_env().deep_model is None


def test_red_alert_drops_builds_already_in_production():
    from examples.llm.redalert import drop_requeued_buildings

    state = {"observation": {"production": [{"queue": "Building", "item": "proc", "progress": 0.4, "ready": False},
                                            {"queue": "Infantry", "item": "e1", "progress": 0.2}]}}
    orders = [{"cmd": "build", "item": "proc"}, {"cmd": "build", "item": "barr"}, {"cmd": "build", "item": "barr"},
              {"cmd": "train", "item": "e1", "count": 3}]
    kept, dropped = drop_requeued_buildings(orders, state)
    assert kept == [{"cmd": "build", "item": "barr"}, {"cmd": "train", "item": "e1", "count": 3}] and dropped == 2


def test_pokemon_plays_on_gpt_6_1_sol_with_claude_as_backup(monkeypatch):
    from examples.llm.providers import FailoverProvider, provider_for_game

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    pokemon = provider_for_game("pokemon_vgc_doubles_draft")
    assert isinstance(pokemon, FailoverProvider)
    assert isinstance(pokemon.primary, OpenAIProvider) and pokemon.primary.model == "gpt-6.1-sol"
    assert isinstance(pokemon.backup, ClaudeProvider)
    red_alert = provider_for_game("red_alert")
    assert isinstance(red_alert, FailoverProvider) and isinstance(red_alert.primary, ClaudeProvider)

    monkeypatch.setenv("POKEMON_PROVIDER", "claude")
    assert isinstance(provider_for_game("pokemon_vgc_doubles_draft").primary, ClaudeProvider)


def test_pokemon_without_an_openai_key_stays_on_claude(monkeypatch):
    from examples.llm.providers import provider_for_game

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(provider_for_game("pokemon_vgc_doubles_draft"), ClaudeProvider)


def test_openai_rate_limits_fail_over_to_claude():
    from examples.llm.providers import FailoverProvider

    limited = ProviderError("OpenAI request failed (HTTP 429, rate_limit_exceeded)")
    provider = FailoverProvider(_Stub("gpt-6.1-sol", limited), _Stub("claude", {"a": 1}), log=lambda _m: None)
    assert provider.complete_structured([], "x", {}) == {"a": 1}


def test_router_builds_the_pokemon_agent_on_gpt(monkeypatch):
    from examples.llm.providers import FailoverProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    context = DecisionContext(session_id="s", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                              agent_id="a", seat_position=0)
    player = Agent()._player(context)
    assert isinstance(player._provider, FailoverProvider) and player._provider.primary.model == "gpt-6.1-sol"
