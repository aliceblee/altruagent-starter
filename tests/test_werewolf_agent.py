"""Tests for the Werewolf agent (agent/werewolf/) and its offline simulator (sim/)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from altruagent import TERMINATE_MESSAGING, DecisionContext, SendMessage
from altruagent.models import GameState
from agent.agent import Agent, create_agent
from agent.werewolf import WerewolfAgent
from agent.werewolf.beliefs import infer
from agent.werewolf.chat_llm import make_polisher
from agent.werewolf.parsing import ChatFacts, Death, Resolver, Snapshot, parse_dead, parse_private, parse_votes, read_message, snapshot
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


def test_seer_claims_its_found_wolf_on_day2_and_messages_respect_limits():
    # (On day 1 the seer stays hidden; see the day-1 seer tests below.)
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, "Your role: seer. Night 1: you investigated Player4: wolf.", phase="messaging", day=2,
              alive=[0, 1, 2, 3, 4, 5], dead=[{"player": "Player6", "role": "villager", "cause": "night_kill", "day": 2}])
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
        content = messages[1]["content"]
        assert "wolf" not in content.lower() or content.startswith(("Draft", "Example"))
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
def test_day1_first_message_is_a_shared_opener_for_every_role(observation):
    from agent.werewolf.agent import DAY1_OPENERS

    agent = WerewolfAgent(rng=random.Random(0))  # no LLM: the line as written
    s = state(2, observation, phase="messaging")
    first = agent.choose_message(s, ctx(2))
    assert isinstance(first, SendMessage) and first.content in DAY1_OPENERS and not first.recipients
    second = agent.choose_message(s, ctx(2))
    assert not (isinstance(second, SendMessage) and second.content in DAY1_OPENERS)


def test_the_opener_varies_from_game_to_game():
    from agent.werewolf.agent import DAY1_OPENERS

    openers = {WerewolfAgent(rng=random.Random(i)).choose_message(state(2, "Your role: villager.", phase="messaging"),
                                                                 ctx(2)).content for i in range(20)}
    assert len(openers) >= 3 and openers <= set(DAY1_OPENERS)
    assert not any("seer" in o.lower() for o in DAY1_OPENERS)  # never points the wolves at the seer


@pytest.mark.parametrize("observation", [
    "Your role: villager.",
    "Your role: seer. Night 1: you investigated Player4: wolf.",
    "Your role: seer. Night 1: you investigated Player4: villager.",
    "Your role: wolf. Your fellow wolf is Player0.",
])
def test_day1_with_nothing_to_go_on_every_role_says_the_same_things(observation):
    # Other agents may keep notes across games: no day-1 sentence of ours may belong to one role.
    from agent.werewolf.agent import DAY1_ASKS, DAY1_OPENERS

    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, observation, phase="messaging")
    texts = [m.content for m in (agent.choose_message(s, ctx(2)) for _ in range(4))
             if isinstance(m, SendMessage) and not m.recipients]
    assert texts and all(t in DAY1_OPENERS + DAY1_ASKS for t in texts)


def test_day1_opener_is_sent_word_for_word_even_with_an_llm():
    from agent.werewolf.agent import DAY1_OPENERS

    seen = []

    def polish(text, facts):
        seen.append(facts)
        return "Something else entirely."

    agent = WerewolfAgent(rng=random.Random(0), polish=polish)
    first = agent.choose_message(state(2, "Your role: villager.", phase="messaging"), ctx(2))
    assert first.content in DAY1_OPENERS
    assert seen == []


def test_a_failing_llm_falls_back_to_the_written_opener():
    from agent.werewolf.agent import DAY1_OPENERS

    def broken(text, facts):
        raise RuntimeError("HTTP 429")

    agent = WerewolfAgent(rng=random.Random(0), polish=broken)
    assert agent.choose_message(state(2, "Your role: villager.", phase="messaging"), ctx(2)).content in DAY1_OPENERS


@pytest.mark.parametrize("rewrite", [
    "I'm the seer and a vegetarian vampire, wolves raise your hand!",  # invents a seer claim
    "Vegetarian vampire here. Player3, raise your hand!",  # names a player
])
def test_opener_rewrites_that_break_the_rules_are_rejected(rewrite):
    from agent.werewolf.agent import DAY1_OPENERS

    out = make_polisher(FakeProvider(rewrite))(DAY1_OPENERS[0], {"intent": "opener"})
    assert out == DAY1_OPENERS[0]


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


# -- regressions from live test matches ------------------------------------------------------

# Day-1 public chat of Testing match ac3bdf0c (all seats ours except 5 and 6):
# P2 was the seer and found P0; wolves P0/P1. Our villagers voted out the seer,
# because "Player0's seer claim makes sense" counted as P1 claiming seer.
AC3BDF0C_DAY1 = [
    (2, "I am the Seer. My checks: Player0 = WOLF. I'm voting Player0."),
    (1, "Villager here, no info. Seer should speak up. I'm voting Player2. Their play looks off to me."),
    (0, "I am the real Seer. I checked Player2: WOLF. They're faking to frame us. I'm voting Player2."),
    (1, "Villager here. Player0's seer claim makes sense to me. I'm voting Player2."),
    (2, "I'm the Seer. Player1 pushing me without evidence is exactly what a wolf would do. I'm voting Player0."),
    (0, "Villager here. I'm voting Player2. Their seer claim doesn't add up."),
    (4, "Noted Player2's claim. I'm voting Player2."),
]


def test_talking_about_someones_seer_claim_is_not_a_seer_claim():
    facts = ChatFacts()
    for text in ("Player0's seer claim makes sense to me.", "Player4's seer claim is credible.",
                 "I'm voting Player1. Their seer claim doesn't add up."):
        read_message(facts, 3, text, [], 1, 6, RESOLVER)
    assert facts.seer_claims == {}
    for text in ("Seer here. Player2 is a wolf.", "As the seer, I checked Player2: wolf."):
        f = ChatFacts()
        read_message(f, 3, text, [], 1, 6, RESOLVER)
        assert 3 in f.seer_claims and f.claimed_results[3] == {2: "wolf"}


def test_an_accusation_is_not_a_seer_result():
    facts = ChatFacts()
    read_message(facts, 2, "I am the Seer. My checks: Player0 = WOLF.", [], 1, 6, RESOLVER)
    read_message(facts, 2, "Player1 pushing me without evidence is exactly what a wolf would do.", [], 1, 6, RESOLVER)
    assert facts.claimed_results[2] == {0: "wolf"}


def test_ac3bdf0c_day1_villager_now_suspects_the_reactive_counterclaimer():
    facts = ChatFacts()
    for sender, text in AC3BDF0C_DAY1:
        read_message(facts, sender, text, [], 1, 3, RESOLVER)
    assert sorted(facts.seer_claims) == [0, 2]
    assert facts.named_before_claim[0] == {2}
    p = infer(snap(me=3), facts).p_wolf
    assert max(p, key=p.get) == 0 and p[2] < 0.25


def test_a_counterclaiming_wolf_never_then_says_villager_here():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(0, "Your role: wolf. Your fellow wolf is Player1.", phase="messaging",
              messages=[(2, "I am the Seer. My checks: Player0 = WOLF. I'm voting Player0.")])
    texts = []
    for _ in range(4):
        m = agent.choose_message(s, ctx(0))
        if isinstance(m, SendMessage) and not m.recipients:
            texts.append(m.content)
    claim = next(i for i, t in enumerate(texts) if "real Seer" in t)
    assert not any("Villager here" in t for t in texts[claim:])


def test_no_info_day1_opening_names_no_random_target():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(6, "Your role: villager.", phase="messaging")
    agent.choose_message(s, ctx(6))  # the day-1 opener
    opening = agent.choose_message(s, ctx(6))
    assert isinstance(opening, SendMessage) and "voting" not in opening.content


# -- wolf play ------------------------------------------------------------------------------------

WOLF_OBS = "Your role: wolf. Your fellow wolf is Player1."


def _talk(agent, seat, observation, messages, *, day=1, calls=4, alive=None):
    s = state(seat, observation, phase="messaging", messages=messages, day=day, alive=alive)
    return [m for m in (agent.choose_message(s, ctx(seat)) for _ in range(calls)) if isinstance(m, SendMessage)]


def test_outed_wolf_counterclaims_without_naming_its_accuser_as_a_wolf():
    agent = WerewolfAgent(rng=random.Random(0))
    agent.counter_claim_style = "other"
    sent = _talk(agent, 0, WOLF_OBS, [(2, "I am the Seer. My checks: Player0 = WOLF. I'm voting Player0.")])
    claim = next(m.content for m in sent if "real Seer" in m.content)
    assert "Player2 is lying" in claim and "Player2: WOLF" not in claim and "Player2 = WOLF" not in claim
    facts = ChatFacts()
    read_message(facts, 2, "I am the Seer. My checks: Player0 = WOLF.", [], 1, 4, RESOLVER)
    read_message(facts, 0, claim, [], 1, 4, RESOLVER)
    assert 2 not in {t for t, v in facts.claimed_results.get(0, {}).items() if v == "wolf"}
    # ...so the reactive-counterclaim penalty doesn't fire against us.
    assert not any(facts.claimed_results.get(0, {}).get(a) == "wolf" for a in facts.named_before_claim[0])


def test_wolf_spares_its_accuser_at_night():
    agent = WerewolfAgent(rng=random.Random(0))
    _talk(agent, 0, WOLF_OBS, [(3, "Player0 is suspicious, I'm voting Player0.")], calls=1)
    night = state(0, WOLF_OBS, sub_phase="night_wolf", legal=[2, 3, 4, 5, 6], day=1)
    assert agent.choose_action(night, ctx(0)).action_id != "3"


def test_endgame_wolf_votes_to_tie_instead_of_losing_its_partner():
    # 5 alive: wolves 0 and 1; villagers 2, 3, 4. Announced: 2->1, 3->1, 4->2. Joining on
    # Player1 loses the partner; a vote on Player2 makes it 2-2, a tie, so nobody goes out.
    agent = WerewolfAgent(rng=random.Random(0))
    talk = [(2, "I'm voting Player1."), (3, "I'm voting Player1."), (4, "I'm voting Player2.")]
    _talk(agent, 0, WOLF_OBS, talk, day=3, alive=[0, 1, 2, 3, 4], calls=1)
    vote = state(0, WOLF_OBS, legal=[1, 2, 3, 4, 7], day=3, alive=[0, 1, 2, 3, 4])
    assert agent.choose_action(vote, ctx(0)).action_id == "2"


def test_endgame_wolf_turns_a_close_vote_onto_a_villager():
    # 5 alive: wolves 0,1; villagers 2,3,4. Announced: 2->3, 4->3, 3->1. Our vote decides nothing
    # against us, so take the villager lynch.
    agent = WerewolfAgent(rng=random.Random(0))
    _talk(agent, 0, WOLF_OBS, [(2, "I'm voting Player3."), (4, "I'm voting Player3."), (3, "I'm voting Player1.")],
          day=2, alive=[0, 1, 2, 3, 4], calls=1)
    vote = state(0, WOLF_OBS, legal=[1, 2, 3, 4, 7], day=2, alive=[0, 1, 2, 3, 4])
    assert agent.choose_action(vote, ctx(0)).action_id == "3"


def test_wolf_spreads_its_vote_when_the_lynch_is_already_decided():
    agent = WerewolfAgent(rng=random.Random(0))
    talk = [(2, "I'm voting Player4."), (3, "I'm voting Player4."), (5, "I'm voting Player4."), (6, "I'm voting Player4.")]
    _talk(agent, 0, WOLF_OBS, talk, calls=1)
    vote = state(0, WOLF_OBS, legal=[1, 2, 3, 4, 5, 6, 7])
    assert agent.choose_action(vote, ctx(0)).action_id not in ("4", "1", "7")


def test_wolf_never_names_its_partner_on_day_1_by_default():
    # A fixed "PlayerX seems a little suspicious" that only our wolves say would point at the other wolf.
    agent = WerewolfAgent(rng=random.Random(0))
    texts = [m.content for m in _talk(agent, 0, WOLF_OBS, []) if not m.recipients]
    assert texts and not any("Player1" in t for t in texts)


@pytest.mark.parametrize("text, kill", [
    ("Partner: vote Player3 today, kill Player5 tonight.", 5),
    ("Wolf plan: we push Player3 today; tonight we kill Player6.", 6),
    ("target 4", 4),
    ("let's eat P2 tonight", 2),
])
def test_partner_kill_suggestion_is_read_loosely(text, kill):
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(0, WOLF_OBS, phase="messaging", messages=[])
    s.raw["new_messages"] = [{"seq": 1, "sender": 1, "recipients": [0], "content": text, "type": "chat"}]
    agent.choose_message(GameState.from_mcp_state(s.raw), ctx(0))
    night = state(0, WOLF_OBS, sub_phase="night_wolf", legal=[2, 3, 4, 5, 6])
    assert agent.choose_action(night, ctx(0)).action_id == str(kill)


def test_wolf_stops_whispering_to_a_silent_partner_after_day_1():
    agent = WerewolfAgent(rng=random.Random(0))
    day2 = _talk(agent, 0, WOLF_OBS, [], day=2)
    assert not any(m.recipients for m in day2)


# Testing match e1b77dea: wolf P3 fake-claimed "Player0 = WOLF" on day 1, P0 was voted out
# and revealed as the real seer, and on day 2 P3 still said "Seer again. My checks:
# Player0 = WOLF, Player1 = villager. I'm voting Player4." Its partner P2 kept backing it.
E1B77DEA_DEAD = [{"player": "Player0", "role": "seer", "cause": "vote", "day": 1},
                 {"player": "Player6", "role": "villager", "cause": "night_kill", "day": 2}]


def _e1b77dea_wolf(seat: int, partner: int) -> WerewolfAgent:
    agent = WerewolfAgent(rng=random.Random(0))
    obs = f"Your role: wolf. Your fellow wolf is Player{partner}."
    day1 = [(0, "I am the Seer. My checks: Player3 = WOLF. I'm voting Player3.")]
    if seat == 2:
        day1.append((3, "I am the real Seer. I checked Player0: WOLF. I'm voting Player0."))
    _talk(agent, seat, obs, day1)
    return agent


def test_a_disproved_fake_seer_drops_the_act():
    agent = _e1b77dea_wolf(3, 2)
    assert agent._fake_seer
    s = state(3, "Your role: wolf. Your fellow wolf is Player2.", phase="messaging", day=2,
              alive=[1, 2, 3, 4, 5], dead=E1B77DEA_DEAD)
    texts = [m.content for m in (agent.choose_message(s, ctx(3)) for _ in range(4))
             if isinstance(m, SendMessage) and not m.recipients]
    assert texts and not any("Seer" in t or "Player0" in t or "Villager here" in t for t in texts)


def test_partner_stops_backing_a_disproved_seer_claim():
    agent = _e1b77dea_wolf(2, 3)
    s = state(2, "Your role: wolf. Your fellow wolf is Player3.", phase="messaging", day=2,
              alive=[1, 2, 3, 4, 5], dead=E1B77DEA_DEAD)
    texts = [m.content for m in (agent.choose_message(s, ctx(2)) for _ in range(4))
             if isinstance(m, SendMessage) and not m.recipients]
    assert not any("seer claim makes sense" in t for t in texts)


def test_a_fake_seer_always_votes_for_a_wolf_it_names():
    # Our claim still stands (nothing revealed contradicts it): whoever we vote for on
    # day 2 must be one of the players we call a wolf in the same message.
    agent = WerewolfAgent(rng=random.Random(0))
    _talk(agent, 0, WOLF_OBS, [(2, "I am the Seer. My checks: Player0 = WOLF.")])
    s = state(0, WOLF_OBS, phase="messaging", day=2, alive=[0, 1, 2, 3, 4, 5],
              dead=[{"player": "Player6", "role": "villager", "cause": "night_kill", "day": 2}])
    claim = next(m.content for m in (agent.choose_message(s, ctx(0)) for _ in range(4))
                 if isinstance(m, SendMessage) and "My checks" in m.content)
    voted = int(claim.rsplit("voting Player", 1)[1].split(".")[0])
    assert f"Player{voted} = WOLF" in claim


@pytest.mark.parametrize("seed", range(20))
def test_whisper_never_plans_to_vote_and_kill_the_same_player(seed):
    agent = WerewolfAgent(rng=random.Random(seed))
    talk = [(5, "Partner: vote Player1 today, kill Player1 tonight.")]
    s = state(0, "Your role: wolf. Your fellow wolf is Player5.", phase="messaging", messages=[])
    s.raw["new_messages"] = [{"seq": 1, "sender": 5, "recipients": [0], "content": talk[0][1], "type": "chat"}]
    whispers = [m for m in (agent.choose_message(GameState.from_mcp_state(s.raw), ctx(0)) for _ in range(3))
                if isinstance(m, SendMessage) and m.recipients]
    assert whispers
    text = whispers[0].content
    import re as _re
    seats = [int(x) for x in _re.findall(r"Player(\d)", text)]
    assert len(seats) == 2 and seats[0] != seats[1], text


# -- day 1 with no information (villager) ------------------------------------------------------

def _villager_day1(messages, seed=0):
    agent = WerewolfAgent(rng=random.Random(seed))
    s = state(6, "Your role: villager.", phase="messaging", messages=messages)
    sent = [m for m in (agent.choose_message(s, ctx(6)) for _ in range(3)) if isinstance(m, SendMessage)]
    vote = agent.choose_action(state(6, "Your role: villager.", legal=[0, 1, 2, 3, 4, 5, 7]), ctx(6))
    return [m.content for m in sent[1:]], vote.action_id  # skip the opener


def test_day1_nobody_voted_yet_asks_for_reads_without_naming_a_random_suspect():
    texts, vote = _villager_day1([])
    assert texts == ["No leads yet. Who's your top suspect, and why? I'll back the strongest case so we don't split."]
    assert vote not in ("6", "7")  # still never abstains
    facts = ChatFacts()
    read_message(facts, 6, texts[0], [], 1, 0, RESOLVER)
    assert facts.intents_on(1) == {}  # not a vote on anyone


def test_day1_a_vote_on_me_gets_called_out_and_voted():
    texts, vote = _villager_day1([(4, "I'm voting Player6.")])
    assert texts[0] == "Player4, you're being really suspicious, quick to point fingers already. I think you're the werewolf."
    assert vote == "4"


def test_day1_joins_the_leading_announced_vote():
    texts, vote = _villager_day1([(1, "I'm voting Player3."), (2, "I'm voting Player5."), (4, "Voting Player3.")])
    assert texts[0] == "Okay, joining Player1's vote on Player3 so we don't split."
    assert vote == "3"


def test_day1_a_credible_seer_claim_decides_the_vote():
    agent = WerewolfAgent(rng=random.Random(0))
    agent.choose_message(state(6, "Your role: villager.", phase="messaging"), ctx(6))
    agent.choose_message(state(6, "Your role: villager.", phase="messaging"), ctx(6))  # asks for reads
    seer = state(6, "Your role: villager.", phase="messaging",
                 messages=[(2, "I am the Seer. My checks: Player4 = WOLF. I'm voting Player4.")])
    agent.choose_message(seer, ctx(6))
    vote = agent.choose_action(state(6, "Your role: villager.", legal=[0, 1, 2, 3, 4, 5, 7]), ctx(6))
    assert vote.action_id == "4"


# -- regressions from the Oct 12 Testing tournament (games eda821fb, 0eafa068, 9c774f95) -----------

def test_an_announced_vote_sticks_through_small_wobbles():
    # A seer with no wolf found announced Player0, then kept "changing" and finally voted
    # someone it never announced. Small shifts in the odds must not move the vote.
    agent = WerewolfAgent(rng=random.Random(0))
    obs = "Your role: villager."
    agent.choose_message(state(6, obs, phase="messaging"), ctx(6))
    agent.choose_message(state(6, obs, phase="messaging", messages=[(1, "I'm voting Player3.")]), ctx(6))
    announced = agent._announced_target
    assert announced == 3
    noise = [(1, "Player2 seems a bit off."), (3, "I trust Player4."), (5, "Player0 is quiet.")]
    for _ in range(3):
        agent.choose_message(state(6, obs, phase="messaging", messages=noise), ctx(6))
    vote = agent.choose_action(state(6, obs, legal=[0, 1, 2, 3, 4, 5, 7]), ctx(6))
    assert vote.action_id == str(announced)


def test_an_llm_rewrite_that_hides_the_vote_gets_it_back():
    agent = WerewolfAgent(rng=random.Random(0), polish=lambda text, facts: "We should go after Player3 today, friends.")
    agent._resolver = RESOLVER
    out = agent._with_readable_vote("We should go after Player3 today, friends.", 3)
    assert out.endswith("I'm voting Player3.")
    assert agent._with_readable_vote("I'm convinced Player3 is a wolf.", 3) == "I'm convinced Player3 is a wolf."


def test_day1_joins_a_push_written_without_the_word_vote():
    texts, vote = _villager_day1([(2, "No hard info yet, but I'm convinced Player0 is a wolf based on their reasoning.")])
    assert texts[0] == "Okay, joining Player2's vote on Player0 so we don't split."
    assert vote == "0"


def test_a_victims_retaliation_does_not_frame_the_accuser():
    # Day 1: P0 picked P1 at random; P1 (the seer) hit back at P0, then died at night.
    facts = ChatFacts()
    read_message(facts, 0, "My psychic ability tells me it's Player1. We should vote to avoid a split.", [], 1, 6, RESOLVER)
    read_message(facts, 1, "Player0's baseless attacks are what only wolves do. I'm voting Player0.", [], 1, 6, RESOLVER)
    dead = {1: Death(1, "seer", "night_kill", 2)}
    p = infer(snap(dead=dead, alive={0, 2, 3, 4, 5, 6}, day=2), facts).p_wolf
    assert p[0] == pytest.approx(p[2], abs=0.02)


def test_endgame_steering_never_overrides_our_vote_when_it_cannot_change_the_result():
    # eda821fb day 2: wolf P1 (fake seer, accused P5) faced 3 votes; it voted Player3, an
    # arbitrary villager, instead of the Player5 it had announced.
    agent = WerewolfAgent(rng=random.Random(0))
    obs = "Your role: wolf. Your fellow wolf is Player4."
    alive = [1, 3, 4, 5, 6]
    talk = [(5, "I am the Seer. My checks: Player1 = WOLF. I'm voting Player1."),
            (3, "I'm voting Player1."), (6, "I'm voting Player1."), (4, "I'm voting Player5.")]
    _talk(agent, 1, obs, talk, day=2, alive=alive, calls=2)
    vote = agent.choose_action(state(1, obs, legal=[3, 4, 5, 6, 7], day=2, alive=alive), ctx(1))
    assert vote.action_id == "5"


# -- unison pushes ----------------------------------------------------------------------------

def _facts(messages, me=6):
    facts = ChatFacts()
    for sender, text in messages:
        read_message(facts, sender, text, [], 1, me, RESOLVER)
    return facts


def test_two_back_to_back_unprompted_pushes_are_unison():
    facts = _facts([(1, "I'm voting Player3."), (4, "Vote Player3, they seem off.")])
    assert facts.unison == [(1, 4, 3, 1)]


@pytest.mark.parametrize("messages", [
    [(1, "I'm voting Player3."), (4, "Okay, joining Player1's vote on Player3 so we don't split.")],  # following
    [(1, "My vampiric psychic ability tells me it's Player3."), (4, "My psychic sense says it's Player3 too, I'm voting Player3.")],
    [(2, "I am the Seer. My checks: Player3 = WOLF."), (1, "I'm voting Player3."), (4, "I'm voting Player3.")],  # seer-backed
    [(1, "I'm voting Player3."), (0, "hi"), (2, "hello"), (5, "hey"), (4, "I'm voting Player3.")],  # far apart
])
def test_not_unison(messages):
    assert _facts(messages).unison == []


def test_villager_calls_out_a_unison_pair_once():
    agent = WerewolfAgent(rng=random.Random(0))
    talk = [(1, "I'm voting Player3."), (4, "Vote Player3, they seem off.")]
    s = state(6, "Your role: villager.", phase="messaging", messages=talk, day=2,
              dead=[{"player": "Player0", "role": "villager", "cause": "night_kill", "day": 2}],
              alive=[1, 2, 3, 4, 5, 6])
    texts = [m.content for m in (agent.choose_message(s, ctx(6)) for _ in range(4)) if isinstance(m, SendMessage)]
    callouts = [t for t in texts if "in unison" in t]
    assert len(callouts) == 1 and "Player1 looks suspicious" in callouts[0] and "Player4 looks suspicious too" in callouts[0]
    facts = _facts([(6, callouts[0])], me=0)  # other agents read it as suspicion of both
    assert {(6, 1, 1), (6, 4, 1)} <= set(facts.accusations)


def test_wolf_never_calls_out_unison():
    agent = WerewolfAgent(rng=random.Random(0))
    talk = [(2, "I'm voting Player3."), (4, "Vote Player3, they seem off.")]
    s = state(0, WOLF_OBS, phase="messaging", messages=talk, day=2, alive=[0, 1, 2, 3, 4, 5],
              dead=[{"player": "Player6", "role": "villager", "cause": "night_kill", "day": 2}])
    texts = [m.content for m in (agent.choose_message(s, ctx(0)) for _ in range(4)) if isinstance(m, SendMessage)]
    assert not any("in unison" in t for t in texts)


def test_wolf_does_not_echo_its_partners_push_in_unison():
    # Partner Player1 just pushed Player3; we want Player3 too. Our announcement must read as
    # following (which the unison detector ignores), not a second unprompted push.
    agent = WerewolfAgent(rng=random.Random(0))
    agent.distance_partner = False
    s = state(0, WOLF_OBS, phase="messaging", day=2, alive=[0, 1, 2, 3, 4, 5],
              dead=[{"player": "Player6", "role": "villager", "cause": "night_kill", "day": 2}],
              messages=[(1, "I'm voting Player3, they seem off.")])
    agent._wolf_preferred_lynch = lambda snap, villagers: 3
    texts = [m.content for m in (agent.choose_message(s, ctx(0)) for _ in range(3))
             if isinstance(m, SendMessage) and not m.recipients]
    facts = _facts([(1, "I'm voting Player3, they seem off.")] + [(0, t) for t in texts], me=5)
    assert facts.unison == []
    assert any(re.search(r"(?:joining|backing) player1\b.*player3", t, re.I) for t in texts)


# -- regressions from Testing games b4bdb262 (mixed, other teams in seats 3-6) and fd2e7047 -------

def test_endorsing_the_seer_is_not_accusing_them():
    facts = _facts([(2, "As a villager, I support Player0's seer claim and trust their judgment. I'm voting for "
                       "Player6, as Player0 has identified them as a wolf.")])
    assert (2, 0) not in {(a, b) for a, b, _ in facts.accusations}
    assert (2, 0) in {(a, b) for a, b, _ in facts.clears}


def test_quoted_words_are_not_the_speakers_accusation():
    facts = _facts([(3, "Compare P1 and P2 today: both say 'confident in my vote for Player4', nearly word for word.")])
    assert 4 not in {b for _, b, _ in facts.accusations} and facts.intents_on(1) == {}


def test_short_player_names_are_understood():
    facts = _facts([(4, "I think P6 is the wolf here."), (5, "Voting P2.")])
    assert (4, 6) in {(a, b) for a, b, _ in facts.accusations} and facts.intents_on(1)[5] == 2


def test_addressing_players_is_not_accusing_them():
    facts = _facts([(6, "Player3, Player5: one of the two seer claims is a wolf.")])
    assert facts.accusations == []


def test_a_known_innocent_or_a_credible_seer_is_never_called_a_wolf():
    # Our seer had checked Player5 as a villager; Player5 then pushed it. Defend without calling P5 a wolf.
    agent = WerewolfAgent(rng=random.Random(0))
    obs = "Your role: seer. Night 1: you investigated Player5: villager."
    texts = [m.content for m in (agent.choose_message(state(2, obs, phase="messaging", day=2, alive=[1, 2, 3, 5, 6],
             dead=[{"player": "Player0", "role": "villager", "cause": "vote", "day": 1},
                   {"player": "Player4", "role": "villager", "cause": "night_kill", "day": 2}],
             messages=[(5, "Player2 is suspicious, I'm voting Player2.")]), ctx(2)) for _ in range(4))
             if isinstance(m, SendMessage)]
    assert not any("Player5" in t and ("wolf" in t.lower() or "suspicious" in t.lower()) for t in texts)


def test_villager_answers_a_credible_seer_who_pushed_it_with_support():
    agent = WerewolfAgent(rng=random.Random(0))
    talk = [(0, "I am the Seer. My checks: Player6 = WOLF. I'm voting Player6."),
            (0, "As the Seer, Player2's remarks look like wolf behavior.")]
    texts = [m.content for m in (agent.choose_message(state(2, "Your role: villager.", phase="messaging", messages=talk), ctx(2))
             for _ in range(4)) if isinstance(m, SendMessage)]
    assert any("Player0, I'm on your side" in t for t in texts)
    heard = _facts([(2, t) for t in texts], me=5)  # how other agents read us
    assert (2, 0) not in {(a, b) for a, b, _ in heard.accusations}


def test_backing_someones_vote_is_following_not_unison():
    facts = _facts([(3, "I'm voting Player0."), (5, "I'm backing Player3's vote on Player0 to keep us united.")])
    assert facts.unison == []


def test_at_most_one_unison_call_out_a_day():
    agent = WerewolfAgent(rng=random.Random(0))
    talk = [(1, "I'm voting Player3."), (4, "Vote Player3."), (0, "I'm voting Player5."), (2, "Voting Player5.")]
    s = state(6, "Your role: villager.", phase="messaging", messages=talk, day=2, alive=[0, 1, 2, 3, 4, 5, 6],
              dead=[])
    texts = [m.content for m in (agent.choose_message(s, ctx(6)) for _ in range(4)) if isinstance(m, SendMessage)]
    assert sum("in unison" in t for t in texts) == 1


# -- the seer never reveals itself on day 1 ------------------------------------------------------

SEER_FOUND_WOLF = "Your role: seer. Night 1: you investigated Player4: wolf."
SEER_FOUND_VILLAGER = "Your role: seer. Night 1: you investigated Player4: villager."


def _seer_day1_texts(observation, messages, calls=4):
    """The seer's public day-1 messages (the private ally message is checked on its own)."""
    agent = WerewolfAgent(rng=random.Random(0), polish=lambda text, facts: "POLISHED " + text)
    s = state(2, observation, phase="messaging", messages=messages)
    sent = [m for m in (agent.choose_message(s, ctx(2)) for _ in range(calls))
            if isinstance(m, SendMessage) and not m.recipients]
    vote = agent.choose_action(state(2, observation, legal=[0, 1, 3, 4, 5, 6, 7]), ctx(2))
    return agent, [m.content for m in sent], vote.action_id


@pytest.mark.parametrize("observation", [SEER_FOUND_WOLF, SEER_FOUND_VILLAGER])
def test_seer_never_claims_on_day1_even_against_a_rival_claim(observation):
    from agent.werewolf.parsing import _SEER_CLAIM

    rival = [(5, "I am the Seer. I checked Player2: WOLF. Vote Player2.")]
    agent, texts, vote = _seer_day1_texts(observation, rival)
    assert not any("sane seer" in t.lower() for t in texts)  # no seer-only stock line
    assert not any(_SEER_CLAIM.search(t) for t in texts) and not agent._claimed_seer
    assert vote == ("4" if "wolf." in observation else "5")  # found wolf, else the fake seer


def test_seer_with_a_found_wolf_votes_it_on_day1_without_naming_it():
    from agent.werewolf.parsing import _SEER_CLAIM

    agent, texts, vote = _seer_day1_texts(SEER_FOUND_WOLF, [])
    assert not any("Player4" in t for t in texts)
    assert not any(_SEER_CLAIM.search(t) for t in texts) and vote == "4"


def test_seer_with_a_found_wolf_joins_a_push_on_it_but_not_on_others():
    agent, texts, vote = _seer_day1_texts(SEER_FOUND_WOLF, [(1, "I'm voting Player4.")])
    assert any("Player4" in t for t in texts) and vote == "4"
    agent, texts, vote = _seer_day1_texts(SEER_FOUND_WOLF, [(1, "I'm voting Player3.")])
    assert not any("Player3" in t for t in texts) and vote == "4"


def test_seer_under_pressure_on_day1_still_does_not_claim():
    from agent.werewolf.parsing import _SEER_CLAIM

    talk = [(1, "I'm voting Player2."), (3, "Player2 is a wolf, voting Player2."), (5, "Voting Player2 too.")]
    agent, texts, _ = _seer_day1_texts(SEER_FOUND_WOLF, talk, calls=5)
    assert not any(_SEER_CLAIM.search(t) for t in texts) and not agent._claimed_seer


def test_seer_claims_on_day2_as_before():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, SEER_FOUND_WOLF, phase="messaging", day=2, alive=[0, 1, 2, 3, 4, 5],
              dead=[{"player": "Player6", "role": "villager", "cause": "night_kill", "day": 2}])
    texts = [m.content for m in (agent.choose_message(s, ctx(2)) for _ in range(3)) if isinstance(m, SendMessage)]
    assert any(t.startswith("I am the Seer. My checks: Player4 = WOLF") for t in texts)


def test_switching_seer_hides_day1_off_restores_the_day1_claim():
    agent = WerewolfAgent(rng=random.Random(0))
    agent.seer_hides_day1 = False
    s = state(2, SEER_FOUND_WOLF, phase="messaging")
    texts = [m.content for m in (agent.choose_message(s, ctx(2)) for _ in range(3)) if isinstance(m, SendMessage)]
    assert any(t.startswith("I am the Seer. My checks: Player4 = WOLF") for t in texts)


# -- day-1 seer claimant and discreet reads ------------------------------------------------------


def test_village_never_votes_the_lone_day1_seer_claimant():
    agent = WerewolfAgent(rng=random.Random(0))
    msgs = [(3, "I am the seer. I checked Player1: villager."), (2, "Player3 is a wolf, fake seer. Vote Player3.")]
    agent.choose_message(state(6, "Your role: villager.", phase="messaging", messages=msgs), ctx(6))
    vote = agent.choose_action(state(6, "Your role: villager.", legal=[0, 1, 2, 3, 4, 5, 7]), ctx(6))
    assert vote.action_id != "3"


def test_two_day1_claimants_are_not_spared():
    agent = WerewolfAgent(rng=random.Random(0))
    msgs = [(3, "I am the seer. Player1 is a wolf."), (4, "I am the seer. Player3 is a wolf.")]
    agent.choose_message(state(6, "Your role: villager.", phase="messaging", messages=msgs), ctx(6))
    assert agent._spared_claimants(agent._update(state(6, "Your role: villager."), ctx(6))[0]) == set()


def test_asked_for_an_opinion_shares_a_read_in_words():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(6, "Your role: villager.", phase="messaging", day=2,
              dead=[{"player": "Player0", "role": "villager", "cause": "night_kill", "day": 1}],
              alive=[1, 2, 3, 4, 5, 6], messages=[(2, "Player6, who do you suspect?")])
    texts = []
    for _ in range(3):
        m = agent.choose_message(s, ctx(6))
        if isinstance(m, SendMessage):
            texts.append(m.content)
    reply = next(t for t in texts if t.startswith("Player2,"))
    assert "my read so far" in reply and "%" not in reply and "voting" in reply


def test_accused_defense_includes_the_read():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(6, "Your role: villager.", phase="messaging", day=2,
              dead=[{"player": "Player0", "role": "villager", "cause": "night_kill", "day": 1}],
              alive=[1, 2, 3, 4, 5, 6], messages=[(2, "Player6 is a wolf. I'm voting Player6.")])
    texts = [m.content for m in (agent.choose_message(s, ctx(6)) for _ in range(3)) if isinstance(m, SendMessage)]
    assert any("My read so far" in t for t in texts)


def test_seer_privately_tells_its_night1_villager_and_never_votes_them():
    from agent.werewolf.agent import SEER_ALLY_LINE

    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, SEER_FOUND_VILLAGER, phase="messaging")
    sent = [m for m in (agent.choose_message(s, ctx(2)) for _ in range(4)) if isinstance(m, SendMessage)]
    private = [m for m in sent if m.recipients]
    assert [(m.content, m.recipients) for m in private] == [(SEER_ALLY_LINE, [4])]  # once, word for word
    assert sent[1] is private[0]  # right after the opener
    talk = state(2, SEER_FOUND_VILLAGER, phase="messaging", messages=[(1, "I'm voting Player4."), (3, "Voting Player4.")])
    agent.choose_message(talk, ctx(2))
    assert agent.choose_action(state(2, SEER_FOUND_VILLAGER, legal=[0, 1, 3, 4, 5, 6, 7]), ctx(2)).action_id != "4"


def test_seer_with_a_night1_wolf_sends_no_ally_message():
    agent = WerewolfAgent(rng=random.Random(0))
    s = state(2, SEER_FOUND_WOLF, phase="messaging")
    sent = [m for m in (agent.choose_message(s, ctx(2)) for _ in range(4)) if isinstance(m, SendMessage)]
    assert not any(m.recipients for m in sent)


def test_seer_mildly_accuses_its_night1_wolf_once_the_wolf_pushes_someone():
    from agent.werewolf.agent import SEER_POKES

    agent, texts, vote = _seer_day1_texts(SEER_FOUND_WOLF, [(4, "I'm voting Player1, they seem off.")])
    pokes = [t for t in texts if "Player4" in t]
    assert len(pokes) == 1 and "Player1" in pokes[0] and vote == "4"
    assert any(pokes[0].endswith(p.format(wolf="Player4", target="Player1")) for p in SEER_POKES)
    assert not any("seer" in t.lower() for t in texts[1:])


def test_seer_stays_quiet_about_its_wolf_while_the_wolf_says_nothing_suspicious():
    agent, texts, vote = _seer_day1_texts(SEER_FOUND_WOLF, [(4, "Hello everyone, any leads?")])
    assert not any("Player4" in t for t in texts) and vote == "4"


def test_curly_apostrophes_read_like_straight_ones():
    # OpenAI models write "I’m": the parser missed their seer claims until this was normalised.
    facts = ChatFacts()
    read_message(facts, 1, "I’m the Seer, counterclaiming Player6. I checked Player6: wolf.", [], 2, 0, RESOLVER)
    assert facts.seer_claims == {1: 2} and facts.claimed_results[1] == {6: "wolf"}
