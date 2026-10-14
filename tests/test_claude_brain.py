"""Tests for the Claude-driven Werewolf brain (agent/werewolf/claude_brain.py), with a fake client."""

from __future__ import annotations

import json
from types import SimpleNamespace

from altruagent import TERMINATE_MESSAGING, SendMessage
from agent.agent import Agent
from agent.werewolf import WerewolfAgent
from agent.werewolf.claude_brain import ClaudeWerewolfAgent, claude_agent_from_env
from tests.test_werewolf_agent import ctx, state


class FakeClient:
    """Answers each request with the next queued answer (a dict, an exception, or a response)."""

    def __init__(self, *answers, fallbacks_ok: bool = True):
        self.answers = list(answers)
        self.requests: list[dict] = []
        self.fallbacks_ok = fallbacks_ok
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._beta_create))
        self.messages = SimpleNamespace(create=self._create)

    def _beta_create(self, **request):
        if not self.fallbacks_ok:
            err = Exception("fallbacks: unknown parameter")
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
        return SimpleNamespace(stop_reason="end_turn", usage=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(answer))])


def brain(client, **kw):
    return ClaudeWerewolfAgent(client, advisor=WerewolfAgent(), playbook="PLAYBOOK TEXT", **kw)


def vote_state(**kw):
    return state(6, "You are Player6. Your role: villager.", legal=[0, 1, 2, 3, 4, 5, 7], **kw)


def chat_state(me=6, observation="You are Player6. Your role: villager.", **kw):
    return state(me, observation, phase="messaging", **kw)


def test_claude_picks_the_vote():
    client = FakeClient({"notes": "Player3 dodged", "action_id": "3"})
    b = brain(client)
    assert b.choose_action(vote_state(), ctx(6)).action_id == "3"
    assert b.notes == "Player3 dodged"
    request = client.requests[0]
    assert "PLAYBOOK TEXT" in request["system"][0]["text"]
    assert request["output_config"]["format"]["schema"]["properties"]["action_id"]["enum"] == ["0", "1", "2", "3", "4", "5", "7"]
    assert "Your role: villager" in request["messages"][0]["content"]
    assert "P(wolf)" in request["messages"][0]["content"]


def test_notes_are_fed_back_next_time():
    client = FakeClient({"notes": "remember Player2", "action_id": "2"}, {"notes": "", "action_id": "2"})
    b = brain(client)
    b.choose_action(vote_state(), ctx(6))
    b.choose_action(vote_state(), ctx(6))
    assert "remember Player2" in client.requests[1]["messages"][0]["content"]


def test_api_failure_falls_back_to_the_rule_engine():
    b = brain(FakeClient(TimeoutError("slow")))
    move = b.choose_action(vote_state(), ctx(6))
    assert move.action_id in {"0", "1", "2", "3", "4", "5"}  # the rules never abstain


def test_refusal_falls_back():
    refusal = SimpleNamespace(stop_reason="refusal", content=[], usage=None)
    move = brain(FakeClient(refusal)).choose_action(vote_state(), ctx(6))
    assert move.action_id != "7"


def test_without_server_fallbacks_it_retries_on_the_plain_endpoint():
    client = FakeClient({"notes": "", "action_id": "4"}, fallbacks_ok=False)
    assert brain(client).choose_action(vote_state(), ctx(6)).action_id == "4"


def test_chat_message_and_private_recipient():
    client = FakeClient({"notes": "", "decision": "send", "recipient": "Player2", "message": "Vote Player4 with me."})
    msg = brain(client).choose_message(
        chat_state(me=1, observation="You are Player1. Your role: wolf. Your fellow wolf is Player2.", day=2), ctx(1))
    assert isinstance(msg, SendMessage) and msg.recipients == [2] and msg.content == "Vote Player4 with me."


def test_chat_end():
    client = FakeClient({"notes": "", "decision": "end", "recipient": "everyone", "message": ""})
    b = brain(client)
    assert b.choose_message(chat_state(day=2), ctx(6)) is TERMINATE_MESSAGING
    assert b.choose_message(chat_state(day=2), ctx(6)) is TERMINATE_MESSAGING  # no second call today
    assert len(client.requests) == 1


def test_seer_never_claims_on_day_1():
    client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone", "message": "I am the seer, Player3 is a wolf."})
    msg = brain(client).choose_message(chat_state(observation="You are Player6. Your role: seer."), ctx(6))
    assert not (isinstance(msg, SendMessage) and "seer" in msg.content.lower() and "I am the seer" in msg.content)


def test_long_messages_are_cut():
    client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone", "message": "word " * 80})
    msg = brain(client).choose_message(chat_state(day=2), ctx(6))
    assert len(msg.content.split()) <= 45


def test_from_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert claude_agent_from_env() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("WEREWOLF_BRAIN", "rules")
    assert claude_agent_from_env() is None
    monkeypatch.delenv("WEREWOLF_BRAIN")
    assert isinstance(claude_agent_from_env(), ClaudeWerewolfAgent)


def test_router_uses_claude_when_keyed_and_rules_when_asked(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("WEREWOLF_BRAIN", raising=False)
    assert isinstance(Agent()._player(ctx(6)), ClaudeWerewolfAgent)
    assert isinstance(Agent(brain="rules")._player(ctx(6)), WerewolfAgent)
