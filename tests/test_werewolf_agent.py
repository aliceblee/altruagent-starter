"""Tests for the Werewolf agent (agent/werewolf/) and its offline simulator (sim/)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from altruagent import TERMINATE_MESSAGING, DecisionContext, SendMessage
from altruagent.models import GameState
from agent.agent import Agent, create_agent
from agent.werewolf import WerewolfAgent
from agent.werewolf.beliefs import infer
from agent.werewolf.chat_llm import make_polisher
from agent.werewolf.parsing import ChatFacts, Resolver, Snapshot, parse_dead, parse_private, parse_votes, read_message, snapshot
from sim.werewolf_sim import BASELINES, Game, ours, run_matchup

PLAYERS = [{"position": i, "name": f"Player{i}", "agent_id": f"a{i}"} for i in range(7)]
RESOLVER = Resolver(PLAYERS, 7)


def ctx(seat: int) -> DecisionContext:
    return DecisionContext(session_id="w", tournament_id=None, game_type="werewolf", agent_id=f"a{seat}", seat_position=seat)


def state(
    me: int,
    observation: str,
    *,
    sub_phase: str = "day_vote",
    legal: list[int] | None = None,
    phase: str = "moving",
    alive: list[int] | None = None,
    dead: list[dict] | None = None,
    votes: list[dict] | None = None,
    messages: list[tuple[int, str]] = (),
    day: int = 1,
) -> GameState:
    alive = list(range(7)) if alive is None else alive
    raw = {
        "session_id": "w",
        "game_type": "werewolf",
        "phase": phase,
        "is_current_actor": legal is not None,
        "state_version": 3,
        "observation": observation,
        "your_position": me,
        "players": PLAYERS,
        "eliminated": False,
        "game_state": {"phase": sub_phase, "day": day, "alive": [f"Player{i}" for i in alive], "dead": dead or [],
                       "vote_history": votes or []},
        "new_messages": [{"seq": n + 1, "sender": s, "recipients": [], "content": text, "type": "chat"}
                         for n, (s, text) in enumerate(messages)],
    }
    if legal is not None:
        raw["legal_actions"] = {"state_version": 3, "actions": [{"action_id": str(a), "label": f"Player{a}", "input": {}} for a in legal]}
    return GameState.from_mcp_state(raw)


def snap(me: int = 6, role: str | None = "villager", **kw) -> Snapshot:
    base = dict(me=me, n_players=7, names={}, alive=set(range(7)), dead={}, role=role, partners=set(), seer_results={},
                day=1, sub_phase="day_vote", votes=[])
    base.update(kw)
    return Snapshot(**base)


# -- parsing ------------------------------------------------------------------------------------


@pytest.mark.parametrize("observation, role", [
    ("You are Player2. Your role: villager.", "villager"),
    ("You are Player2. Your role is Werewolf.", "wolf"),
    ("You are the seer.", "seer"),
    ("role: WOLF", "wolf"),
])
def test_role_is_read_from_varied_wordings(observation, role):
    assert parse_private({}, observation, 2, RESOLVER)[0] == role


@pytest.mark.parametrize("observation", [
    "You are Player2. Your role: wolf. Your fellow wolf is Player5.",
    "Your role: werewolf. The wolves are Player2 and Player5.",
    "Your role: wolf. Partner: Player 5",
])
def test_wolf_partner_is_read(observation):
    assert parse_private({}, observation, 2, RESOLVER)[1] == {5}


def test_wolf_partner_ignores_event_sentences():
    obs = "Your role: wolf. Your teammate is Player4. Player1 was killed by the wolves."
    assert parse_private({}, obs, 2, RESOLVER)[1] == {4}


@pytest.mark.parametrize("observation, expected", [
    ("Your role: seer. Night 1: you investigated Player3: wolf.", {3: "wolf"}),
    ("Your role: seer. You investigated Player3. Player3 is a villager.", {3: "villager"}),
    ("Your role: seer. Investigation results: Player1 -> not a wolf; Player4 -> werewolf", {1: "villager", 4: "wolf"}),
])
def test_seer_results_are_read(observation, expected):
    assert parse_private({}, observation, 2, RESOLVER)[2] == expected


def test_structured_private_fields_win():
    raw = {"private": {"role": "seer", "investigations": {"Player4": "wolf"}}}
    role, partners, results = parse_private(raw, "", 2, RESOLVER)
    assert (role, results) == ("seer", {4: "wolf"})


def test_dead_and_vote_history_shapes():
    dead = parse_dead({"dead": [{"name": "Player3", "role": "seer", "cause": "night_kill", "day": 2}, "Player1"]}, RESOLVER)
    assert dead[3].role == "seer" and dead[3].cause == "night_kill" and dead[1].role is None
    votes = parse_votes([{"day": 1, "votes": {"Player0": "Player3", "Player1": "abstain"}},
                         {"day": 2, "voter": "Player2", "target": 7}], RESOLVER)
    assert [(v.day, v.voter, v.target) for v in votes] == [(1, 0, 3), (1, 1, None), (2, 2, None)]


def test_display_names_are_resolved():
    resolver = Resolver([{"position": 0, "name": "Alpha"}, {"position": 1, "name": "Beta"}], 7)
    facts = ChatFacts()
    read_message(facts, 1, "I am the seer. Alpha is a wolf, vote Alpha.", [], 1, 6, resolver)
    assert facts.claimed_results == {1: {0: "wolf"}} and facts.intents_on(1) == {1: 0}


def test_chat_claims_accusations_and_negation():
    facts = ChatFacts()
    read_message(facts, 4, "I'm the Seer. I checked Player2: wolf. Player5 is not a wolf.", [], 1, 6, RESOLVER)
    read_message(facts, 1, "I'm voting Player4, they seem suspicious.", [], 1, 6, RESOLVER)
    read_message(facts, 3, "I am not the seer, just a villager.", [], 1, 6, RESOLVER)
    assert facts.seer_claims == {4: 1}
    assert facts.claimed_results[4] == {2: "wolf", 5: "villager"}
    assert facts.intents_on(1) == {1: 4}
    assert 3 in facts.villager_claims and 3 not in facts.seer_claims


# -- beliefs ------------------------------------------------------------------------------------


def test_a_lone_seer_claim_makes_its_wolf_target_the_top_suspect():
    facts = ChatFacts()
    read_message(facts, 5, "I am the seer. Player0 is a wolf.", [], 1, 6, RESOLVER)
    p = infer(snap(), facts).p_wolf
    assert max(range(6), key=p.get) == 0
    assert p[5] < p[0]


def test_seer_knowledge_is_certain():
    p = infer(snap(me=2, role="seer", seer_results={4: "wolf", 1: "villager"}), ChatFacts()).p_wolf
    assert p[4] == pytest.approx(1.0) and p[1] == pytest.approx(0.0) and p[2] == 0.0


def test_a_rival_seer_claim_is_a_wolf_to_the_real_seer():
    facts = ChatFacts()
    read_message(facts, 3, "I am the seer, Player2 is a wolf.", [], 1, 2, RESOLVER)
    p = infer(snap(me=2, role="seer", seer_results={0: "villager"}), facts).p_wolf
    assert p[3] > 0.85 and p[3] == max(p.values())


# -- decisions ----------------------------------------------------------------------------------


def test_seer_votes_its_known_wolf():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, "You are Player2. Your role: seer. Night 1: you investigated Player4: wolf.", legal=[0, 1, 3, 4, 5, 6, 7])
    assert agent.choose_action(s, ctx(2)).action_id == "4"


def test_wolf_never_votes_its_partner_by_default_and_never_abstains():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, "Your role: wolf. Your fellow wolf is Player0.", legal=[0, 1, 3, 4, 5, 6, 7])
    assert agent.choose_action(s, ctx(2)).action_id not in ("0", "7")


def test_wolf_kills_the_seer_claimant():
    agent = WerewolfAgent(rng=random.Random(0))
    talk = state(2, "Your role: wolf. Your fellow wolf is Player0.", phase="messaging",
                 messages=[(5, "I am the seer. Player3 is a villager.")])
    agent.choose_message(talk, ctx(2))
    night = state(2, "Your role: wolf. Your fellow wolf is Player0.", sub_phase="night_wolf", legal=[1, 3, 4, 5, 6])
    assert agent.choose_action(night, ctx(2)).action_id == "5"


def test_role_and_partner_come_from_the_night_actions_when_the_observation_is_unreadable(capsys):
    agent = WerewolfAgent(rng=random.Random(0))
    night = state(2, "(something unexpected)", sub_phase="night_wolf", legal=[1, 3, 4, 5, 6])
    agent.choose_action(night, ctx(2))
    assert "WARNING" in capsys.readouterr().out
    vote = state(2, "(something unexpected)", legal=[0, 1, 3, 4, 5, 6, 7])
    assert agent.choose_action(vote, ctx(2)).action_id != "0"  # Player0 is the inferred partner


def test_seer_claims_when_it_found_a_wolf_and_messages_respect_limits():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, "Your role: seer. Night 1: you investigated Player4: wolf.", phase="messaging")
    agent.choose_message(s, ctx(2))  # the day-1 opener
    message = agent.choose_message(s, ctx(2))
    assert isinstance(message, SendMessage) and "Seer" in message.content and "Player4" in message.content
    assert len(message.content.split()) <= 45 and not message.recipients


def test_chat_ends_once_nothing_new_happens():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(6, "Your role: villager.", phase="messaging")
    assert isinstance(agent.choose_message(s, ctx(6)), SendMessage)  # the day-1 opener
    assert isinstance(agent.choose_message(s, ctx(6)), SendMessage)
    assert agent.choose_message(s, ctx(6)) is TERMINATE_MESSAGING
    assert agent.choose_message(s, ctx(6)) is TERMINATE_MESSAGING


def test_wolf_whispers_to_its_partner_once_a_day():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, "Your role: wolf. Your fellow wolf is Player0.", phase="messaging")
    agent.choose_message(s, ctx(2))  # the day-1 opener
    first = agent.choose_message(s, ctx(2))
    assert isinstance(first, SendMessage) and first.recipients == [0]
    second = agent.choose_message(s, ctx(2))
    assert isinstance(second, SendMessage) and not second.recipients


# -- LLM polish ---------------------------------------------------------------------------------


class FakeProvider:
    def __init__(self, text):
        self.text = text

    def complete_structured(self, messages, schema_name, schema, timeout=None):
        assert "wolf" not in messages[1]["content"].lower() or "Draft" in messages[1]["content"]
        return {"message": self.text}


@pytest.mark.parametrize("rewrite, accepted", [
    ("Trust me, Player3 is our best lynch today. I'm voting Player3.", True),
    ("Let's all vote Player4.", False),  # new player, dropped target
    ("I am the seer and I'm voting Player3.", False),  # invented a seer claim
])
def test_polish_keeps_the_plan(rewrite, accepted):
    draft = "Villager here. I'm voting Player3. Their play looks off to me."
    out = make_polisher(FakeProvider(rewrite))(draft, {"intent": "vote", "vote": 3})
    assert out == (rewrite if accepted else draft)


# -- router & simulator -------------------------------------------------------------------------


def test_router_sends_werewolf_to_the_werewolf_agent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = create_agent()
    assert isinstance(agent, Agent)
    s = state(2, "Your role: villager.", legal=[0, 1, 3, 4, 5, 6, 7])
    agent.choose_action(s, ctx(2))
    assert isinstance(agent._delegate, WerewolfAgent)


@pytest.mark.parametrize("baseline", list(BASELINES))
def test_simulated_games_finish_without_errors(baseline):
    wins, _ = run_matchup(ours, BASELINES[baseline], hero_seats=2, games=15, seed=3)
    assert wins["all"][1] == 30


def test_full_self_play_game_finishes():
    rng = random.Random(5)
    result = Game([ours(rng) for _ in range(7)], rng).play()
    assert result.winner in ("wolves", "village")


# -- recorded platform states ----------------------------------------------------------
# Real states from a Testing self-play match, recorded by agent/recorder.py
# (tests/fixtures/werewolf/recorded_states.json): the actual observation wording
# and game_state shapes, display names included.

_RECORDED = json.loads((Path(__file__).parent / "fixtures" / "werewolf" / "recorded_states.json").read_text())


@pytest.mark.parametrize("case", _RECORDED["cases"], ids=[c["name"] for c in _RECORDED["cases"]])
def test_snapshot_reads_recorded_platform_states(case):
    expected = case["expected"]
    snap, _ = snapshot(GameState.from_mcp_state(case["raw"]), case["seat"])
    assert snap.me == case["seat"]
    assert snap.day == expected["day"]
    assert snap.role == expected["role"]
    assert sorted(snap.partners) == expected["partners"]
    assert snap.seer_results == {int(s): v for s, v in expected["seer"].items()}
    assert sorted(snap.alive) == expected["alive"]
    assert {s: [d.role, d.cause] for s, d in snap.dead.items()} == {int(s): v for s, v in expected["dead"].items()}


@pytest.mark.parametrize("observation", [
    "Your role: villager.",
    "Your role: seer. Night 1: you investigated Player4: wolf.",
    "Your role: wolf. Your fellow wolf is Player0.",
])
def test_day1_first_message_is_the_eddie_cullen_opener_for_every_role(observation):
    from agent.werewolf.agent import DAY1_OPENER

    agent = WerewolfAgent(rng=random.Random(0), polish=lambda text, facts: "POLISHED")
    s = state(2, observation, phase="messaging")
    first = agent.choose_message(s, ctx(2))
    assert isinstance(first, SendMessage) and first.content == DAY1_OPENER and not first.recipients
    second = agent.choose_message(s, ctx(2))
    assert not (isinstance(second, SendMessage) and second.content == DAY1_OPENER)


def test_the_opener_is_sent_only_on_day_1():
    agent = WerewolfAgent(rng=random.Random(0))
    message = agent.choose_message(state(2, "Your role: villager.", phase="messaging", day=2), ctx(2))
    assert not (isinstance(message, SendMessage) and "Eddie Cullen" in message.content)


def test_a_false_accusation_raises_the_accusers_wolf_odds_for_a_known_villager():
    facts = ChatFacts()
    read_message(facts, 3, "Player6 is a wolf, I'm voting Player6.", [], 1, 6, RESOLVER)
    from agent.werewolf.beliefs import Weights

    on = Weights(accused_me_falsely=1.4)
    p = infer(snap(), facts, weights=on).p_wolf
    assert p[3] > p[1] and max(range(6), key=p.get) == 3
    assert infer(snap(), facts).p_wolf[3] == pytest.approx(infer(snap(), facts).p_wolf[1], abs=0.05)  # off by default
    # A wolf knows nothing new from it about the accuser.
    wolf_view = infer(snap(role="wolf", partners={0}), facts, weights=on).p_wolf
    assert wolf_view[3] == pytest.approx(wolf_view[1])


def test_rescue_rallies_the_table_when_votes_pile_up():
    agent = WerewolfAgent(rng=random.Random(0))
    talk = [(1, "I'm voting Player2."), (3, "Vote Player2, suspicious."), (4, "I'm voting Player2 too.")]
    s = state(2, "Your role: villager.", phase="messaging", messages=talk)
    sent = [agent.choose_message(s, ctx(2)) for _ in range(3)]  # opener, opening, then the rescue
    texts = [m.content for m in sent if isinstance(m, SendMessage)]
    rescue = next(t for t in texts if t.startswith("Stop:"))
    assert "3 votes on me" in rescue and "villager" in rescue and len(rescue.split()) <= 45
    target = int(rescue.rsplit("vote Player", 1)[1].split()[0])
    assert target != 2
    vote = agent.choose_action(state(2, "Your role: villager.", legal=[0, 1, 3, 4, 5, 6, 7]), ctx(2))
    assert vote.action_id == str(target)


def test_no_rescue_for_a_single_vote():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, "Your role: villager.", phase="messaging", messages=[(1, "I'm voting Player2.")])
    texts = [m.content for m in (agent.choose_message(s, ctx(2)) for _ in range(4)) if isinstance(m, SendMessage)]
    assert not any(t.startswith("Stop:") for t in texts)
