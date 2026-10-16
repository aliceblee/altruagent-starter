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


# -- chat rules enforced in code ------------------------------------------------------------------


def test_day1_opener_is_sent_word_for_word_without_asking_claude():
    from agent.werewolf.agent import DAY1_OPENERS

    client = FakeClient()
    msg = brain(client).choose_message(chat_state(day=1), ctx(6))
    assert msg.content in DAY1_OPENERS and client.requests == []


def _decisions(request):
    return request["output_config"]["format"]["schema"]["properties"]["decision"]["enum"]


def test_claude_may_not_end_while_accused():
    client = FakeClient({"notes": "", "decision": "send", "recipient": "Player2", "message": "Player2, I'm a villager."})
    b = brain(client)
    msg = b.choose_message(chat_state(day=2, messages=[(2, "Player6 is a wolf, I'm voting Player6.")]), ctx(6))
    assert _decisions(client.requests[0]) == ["send"]
    assert "Player2 raised a concern about you" in client.requests[0]["messages"][0]["content"]
    assert isinstance(msg, SendMessage)


def test_claude_may_not_end_when_asked_its_opinion():
    client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone", "message": "Player3 worries me."})
    brain(client).choose_message(chat_state(day=2, messages=[(3, "Player6, what do you think?")]), ctx(6))
    assert _decisions(client.requests[0]) == ["send"]
    assert "Player3 asked you something" in client.requests[0]["messages"][0]["content"]


def test_claude_may_not_end_while_new_messages_arrive_but_may_when_quiet():
    client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone", "message": "Agreed on Player3."},
                        {"notes": "", "decision": "end", "recipient": "everyone", "message": ""})
    b = brain(client)
    b.choose_message(chat_state(day=2, messages=[(3, "I think Player1 is quiet today.")]), ctx(6))
    assert _decisions(client.requests[0]) == ["send"]
    assert b.choose_message(chat_state(day=2), ctx(6)) is TERMINATE_MESSAGING  # nothing new since
    assert _decisions(client.requests[1]) == ["send", "end"]


def test_claude_may_end_when_time_is_nearly_up():
    client = FakeClient({"notes": "", "decision": "end", "recipient": "everyone", "message": ""})
    s = chat_state(day=2, messages=[(2, "Player6 is a wolf.")])
    s.raw["messaging_seconds_left"] = 20
    assert brain(client).choose_message(s, ctx(6)) is TERMINATE_MESSAGING
    assert _decisions(client.requests[0]) == ["send", "end"]


def test_the_seer_does_not_name_its_wolf_on_day1_until_someone_else_does():
    seer = "You are Player6. Your role: seer. Night 1: you investigated Player4: wolf."
    text = "My read so far: Player4 worries me most. I'm voting Player4."
    client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone", "message": text},
                        {"notes": "", "decision": "send", "recipient": "everyone", "message": text})
    b = brain(client)
    b.choose_message(chat_state(observation=seer, day=1), ctx(6))  # the opener
    msg = b.choose_message(chat_state(observation=seer, day=1, messages=[(1, "Any leads?")]), ctx(6))
    assert not (isinstance(msg, SendMessage) and msg.content == text)
    msg = b.choose_message(chat_state(observation=seer, day=1, messages=[(1, "Any leads?"), (2, "I'm voting Player4.")]),
                           ctx(6))
    assert msg.content == text  # someone else accused Player4 first: joining is fine


def test_chat_effort_defaults_to_medium():
    client = FakeClient({"notes": "", "decision": "end", "recipient": "everyone", "message": ""})
    brain(client).choose_message(chat_state(day=2), ctx(6))
    assert client.requests[0]["output_config"]["effort"] == "medium"


def test_claude_is_overruled_when_it_votes_the_lone_day1_seer_claimant():
    client = FakeClient({"notes": "", "action_id": "3"})
    b = brain(client)
    b.choose_message(chat_state(day=1, messages=[(3, "I am the seer. I checked Player1: villager.")]), ctx(6))  # opener
    assert b.choose_action(vote_state(), ctx(6)).action_id != "3"
    assert len(client.requests) == 1  # Claude was asked, and overruled


def test_claude_never_says_a_player_has_been_silent():
    # Our view of the chat can lag: a "silent" player may have spoken while Claude was thinking.
    for text in ("A hunch on Player5, who's been silent so far.", "Player5 hasn't said anything yet.",
                 "We haven't heard from Player4, so I'm voting Player4."):
        client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone", "message": text})
        msg = brain(client).choose_message(chat_state(day=2, messages=[(3, "Hi all.")]), ctx(6))
        assert not (isinstance(msg, SendMessage) and msg.content == text)


def test_claude_must_answer_when_told_it_said_something_wrong():
    client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone",
                         "message": "My mistake, Player5 had spoken. I'm still voting Player5."})
    b = brain(client)
    msg = b.choose_message(chat_state(day=2, messages=[(1, "Calling Player5 silent is inaccurate, P6.")]), ctx(6))
    prompt = client.requests[0]["messages"][0]["content"]
    assert _decisions(client.requests[0]) == ["send"]
    assert "Player1 said something you said was wrong" in prompt and "My mistake" in prompt
    assert msg.content.startswith("My mistake")


def test_a_correction_aimed_at_someone_else_is_not_ours_to_answer():
    client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone", "message": "Agreed."})
    brain(client).choose_message(chat_state(day=2, messages=[(1, "Player2's claim is false.")]), ctx(6))
    assert "said something you said was wrong" not in client.requests[0]["messages"][0]["content"]


def test_owning_up_to_a_silence_claim_is_allowed():
    text = "Player1, fair: calling Player5 silent was my mistake. I'm voting Player1."
    client = FakeClient({"notes": "", "decision": "send", "recipient": "everyone", "message": text})
    assert brain(client).choose_message(chat_state(day=2, messages=[(1, "Hi.")]), ctx(6)).content == text


def _send(text):
    return {"notes": "", "decision": "send", "recipient": "everyone", "message": text}


def test_after_its_messages_run_out_claude_stays_quiet_until_a_seer_claim_then_speaks_once_more():
    # Mixed game 2026-10-15: we used 4 messages in 17 s, went silent, and then voted against our announced
    # target after a seer claim we never answered.
    client = FakeClient(*[_send(f"Message {n}. I'm voting Player5.") for n in range(3)],
                        _send("Player4 claimed seer, so I'm moving my vote off Player5: I'm voting Player1."))
    b = brain(client)
    talk = [(1, "Hi.")]
    b.choose_message(chat_state(day=1, messages=talk), ctx(6))  # the opener
    for n in range(3):
        talk = talk + [(2, f"Thoughts number {n}.")]
        assert isinstance(b.choose_message(chat_state(day=1, messages=talk), ctx(6)), SendMessage)
    talk = talk + [(3, "Anything else?")]
    assert b.choose_message(chat_state(day=1, messages=talk), ctx(6)) is TERMINATE_MESSAGING  # out of messages
    assert len(client.requests) == 3
    talk = talk + [(4, "I am the Seer. I checked Player1: WOLF.")]
    msg = b.choose_message(chat_state(day=1, messages=talk), ctx(6))
    assert isinstance(msg, SendMessage) and "Player1" in msg.content
    assert "Player4 claimed seer" in client.requests[-1]["messages"][0]["content"]
    talk = talk + [(4, "Vote Player1!")]
    assert b.choose_message(chat_state(day=1, messages=talk), ctx(6)) is TERMINATE_MESSAGING  # the server's 5 used
    assert len(client.requests) == 4


def test_after_ending_claude_speaks_again_only_when_accused():
    client = FakeClient({"notes": "", "decision": "end", "recipient": "everyone", "message": ""},
                        _send("Player2, I'm a villager; my vote stays on Player3."))
    b = brain(client)
    assert b.choose_message(chat_state(day=2), ctx(6)) is TERMINATE_MESSAGING
    assert b.choose_message(chat_state(day=2, messages=[(1, "Hello.")]), ctx(6)) is TERMINATE_MESSAGING
    assert len(client.requests) == 1  # nothing important: no Claude call
    msg = b.choose_message(chat_state(day=2, messages=[(1, "Hello."), (2, "Player6 is a wolf, I'm voting Player6.")]),
                           ctx(6))
    assert isinstance(msg, SendMessage) and len(client.requests) == 2


def test_the_vote_request_reminds_claude_of_its_announced_vote():
    client = FakeClient(_send("I'm voting Player3."), {"notes": "", "action_id": "3"})
    b = brain(client)
    b.choose_message(chat_state(day=2), ctx(6))
    b.choose_action(vote_state(day=2, messages=[(6, "I'm voting Player3.")]), ctx(6))
    assert "you announced a vote for Player3" in client.requests[-1]["messages"][0]["content"]


def test_claude_seer_sends_the_ally_message_word_for_word_and_wont_vote_its_ally():
    from agent.werewolf.agent import SEER_ALLY_LINE

    seer = "You are Player6. Your role: seer. Night 1: you investigated Player4: villager."
    client = FakeClient({"notes": "", "action_id": "4"})
    b = brain(client)
    b.choose_message(chat_state(observation=seer, day=1), ctx(6))  # the opener
    msg = b.choose_message(chat_state(observation=seer, day=1, messages=[(1, "Hi.")]), ctx(6))
    assert (msg.content, msg.recipients) == (SEER_ALLY_LINE, [4]) and client.requests == []
    vote = b.choose_action(state(6, seer, legal=[0, 1, 2, 3, 4, 5, 7]), ctx(6))
    assert vote.action_id != "4"


def test_claude_seer_may_pick_on_its_wolf_once_the_wolf_accuses_someone():
    seer = "You are Player6. Your role: seer. Night 1: you investigated Player4: wolf."
    poke = "Player4, that was quick to go after Player1 with nothing to go on. I'm leaning Player4."
    client = FakeClient(_send(poke), _send(poke))
    b = brain(client)
    b.choose_message(chat_state(observation=seer, day=1), ctx(6))  # the opener
    talk = [(4, "Hello, any leads?")]
    msg = b.choose_message(chat_state(observation=seer, day=1, messages=talk), ctx(6))
    assert not (isinstance(msg, SendMessage) and msg.content == poke)  # nothing suspicious yet
    assert "Don't name Player4" in client.requests[0]["messages"][0]["content"]
    talk = talk + [(4, "I'm voting Player1, they seem off.")]
    msg = b.choose_message(chat_state(observation=seer, day=1, messages=talk), ctx(6))
    prompt = client.requests[1]["messages"][0]["content"]
    assert msg.content == poke and "now fair game" in prompt and "they seem off" in prompt
