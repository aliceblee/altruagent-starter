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
def test_day1_first_message_is_the_eddie_cullen_opener_for_every_role(observation):
    from agent.werewolf.agent import DAY1_OPENER

    agent = WerewolfAgent(rng=random.Random(0))  # no LLM: the line as written
    s = state(2, observation, phase="messaging")
    first = agent.choose_message(s, ctx(2))
    assert isinstance(first, SendMessage) and first.content == DAY1_OPENER and not first.recipients
    second = agent.choose_message(s, ctx(2))
    assert not (isinstance(second, SendMessage) and second.content == DAY1_OPENER)


def test_day1_opener_is_rewritten_by_the_llm_when_one_is_configured():
    seen = []

    def polish(text, facts):
        seen.append(facts)
        return "Vegetarian vampire villager here, fangs off. Werewolves, paws up!"

    agent = WerewolfAgent(rng=random.Random(0), polish=polish)
    first = agent.choose_message(state(2, "Your role: villager.", phase="messaging"), ctx(2))
    assert first.content == "Vegetarian vampire villager here, fangs off. Werewolves, paws up!"
    assert seen[0] == {"intent": "opener"}


def test_a_failing_llm_falls_back_to_the_written_opener():
    from agent.werewolf.agent import DAY1_OPENER

    def broken(text, facts):
        raise RuntimeError("HTTP 429")

    agent = WerewolfAgent(rng=random.Random(0), polish=broken)
    assert agent.choose_message(state(2, "Your role: villager.", phase="messaging"), ctx(2)).content == DAY1_OPENER


@pytest.mark.parametrize("rewrite", [
    "I'm the seer and a vegetarian vampire, wolves raise your hand!",  # invents a seer claim
    "Vegetarian vampire here. Player3, raise your hand!",  # names a player
])
def test_opener_rewrites_that_break_the_rules_are_rejected(rewrite):
    from agent.werewolf.agent import DAY1_OPENER

    out = make_polisher(FakeProvider(rewrite))(DAY1_OPENER, {"intent": "opener"})
    assert out == DAY1_OPENER


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
    # 4 alive: wolves 0 and 1, villagers 2 and 3. Both villagers announce Player1.
    agent = WerewolfAgent(rng=random.Random(0))
    _talk(agent, 0, WOLF_OBS, [(2, "I'm voting Player1."), (3, "I'm voting Player1.")], day=3, alive=[0, 1, 2, 3], calls=1)
    vote = state(0, WOLF_OBS, legal=[1, 2, 3, 7], day=3, alive=[0, 1, 2, 3])
    # Player1 gets 2 (+1 from the partner's own vote elsewhere) no matter what we do, so
    # the best we can do is make it a tie or push a villager: never join in on Player1.
    assert agent.choose_action(vote, ctx(0)).action_id != "1"


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


def test_wolf_voices_mild_doubt_about_its_partner_on_day_1():
    agent = WerewolfAgent(rng=random.Random(0))
    texts = [m.content for m in _talk(agent, 0, WOLF_OBS, []) if not m.recipients]
    assert any("Player1 seems a little suspicious" in t for t in texts)


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
