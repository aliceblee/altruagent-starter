"""The router's Werewolf safety net and decision log (agent/agent.py, agent/decision_log.py)."""

from __future__ import annotations

import json

import pytest

from altruagent import TERMINATE_MESSAGING, DecisionContext, SendMessage, WithReasoning
from altruagent.models import GameState
from agent.agent import Agent
from agent.decision_log import DecisionLog

PLAYERS = [{"position": i, "name": f"Player{i}", "agent_id": f"a{i}"} for i in range(7)]


def ctx(seat: int = 2, game_type: str = "werewolf") -> DecisionContext:
    return DecisionContext(session_id="w:1", tournament_id=None, game_type=game_type, agent_id=f"a{seat}", seat_position=seat)


def state(me: int = 2, *, sub_phase: str = "day_vote", legal: list[int] | None = None, alive=range(7), day: int = 1) -> GameState:
    raw = {
        "session_id": "w:1", "game_type": "werewolf", "phase": "moving" if legal else "messaging",
        "is_current_actor": bool(legal), "state_version": 3, "observation": "Your role: VILLAGER",
        "your_position": me, "players": PLAYERS, "eliminated": False, "messaging_seconds_left": 90,
        "game_state": {"phase": sub_phase, "day": day, "alive": [f"Player{i}" for i in alive], "dead": [], "vote_history": []},
    }
    if legal:
        raw["legal_actions"] = {"state_version": 3, "actions": [{"action_id": str(a), "label": f"Player{a}", "input": {}} for a in legal]}
    return GameState.from_mcp_state(raw)


class Scripted:
    """A stand-in WerewolfAgent that returns (or raises) whatever it's given."""

    def __init__(self, action=None, message=None):
        self.action, self.message = action, message

    def choose_action(self, state, context):
        if isinstance(self.action, Exception):
            raise self.action
        return self.action

    def choose_message(self, state, context):
        if isinstance(self.message, Exception):
            raise self.message
        return self.message


def agent(tmp_path, **kw) -> Agent:
    return Agent(delegate=Scripted(**kw), decision_log=DecisionLog(tmp_path))


def log_lines(tmp_path, seat: int = 2) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "w_1" / f"seat{seat}.jsonl").read_text().splitlines()]


DAY_VOTE = [0, 1, 3, 4, 5, 6, 7]


# -- moves ------------------------------------------------------------------------------


@pytest.mark.parametrize("returned", ["4", 4, WithReasoning("4", "why")])
def test_a_legal_move_passes_through_unchanged(tmp_path, returned):
    s = state(legal=DAY_VOTE)
    assert agent(tmp_path, action=returned).choose_action(s, ctx()) is returned
    assert log_lines(tmp_path)[0] | {"time": 0, "ms": 0} == {
        "time": 0, "ms": 0, "kind": "action", "seat": 2, "day": 1, "sub_phase": "day_vote",
        "legal": [str(a) for a in DAY_VOTE], "action_id": "4", "label": "Player4", "source": "agent", "error": None,
        "observation": "Your role: VILLAGER",
    }


def test_a_crash_by_day_abstains_like_the_server_would(tmp_path, capsys):
    a = agent(tmp_path, action=ValueError("boom"))
    assert a.choose_action(state(legal=DAY_VOTE), ctx()).action_id == "7"
    assert "FALLBACK move (day_vote): ValueError: boom" in capsys.readouterr().out
    line = log_lines(tmp_path)[0]
    assert (line["source"], line["action_id"]) == ("fallback", "7")
    assert line["error"].startswith("ValueError: boom at test_agent_safety.py:")


def test_a_crash_at_night_plays_the_first_legal_target(tmp_path):
    a = agent(tmp_path, action=KeyError("x"))
    assert a.choose_action(state(sub_phase="night_wolf", legal=[0, 3, 5]), ctx()).action_id == "0"


@pytest.mark.parametrize("returned", ["2", 9, None, "RESIGN"])
def test_an_illegal_move_is_replaced(tmp_path, returned):
    a = agent(tmp_path, action=returned)
    assert a.choose_action(state(legal=DAY_VOTE), ctx()).action_id == "7"
    assert "not a legal action" in log_lines(tmp_path)[0]["error"]


def test_other_games_are_not_wrapped(tmp_path):
    a = agent(tmp_path, action=ValueError("boom"))
    with pytest.raises(ValueError):
        a.choose_action(state(legal=[0, 1]), ctx(game_type="pokemon_vgc_doubles_draft"))


# -- messages ---------------------------------------------------------------------------


def test_a_valid_chat_passes_through_and_is_logged(tmp_path):
    message = SendMessage("Player4 dodged my question. I'm voting Player4.")
    assert agent(tmp_path, message=message).choose_message(state(), ctx()) is message
    line = log_lines(tmp_path)[0]
    assert (line["kind"], line["type"], line["content"], line["recipients"], line["source"]) == (
        "message", "chat", message.content, [], "agent")
    assert line["seconds_left"] == 90


def test_a_long_chat_is_trimmed_to_45_words(tmp_path):
    sent = agent(tmp_path, message=SendMessage(" ".join(["word"] * 60))).choose_message(state(), ctx())
    assert isinstance(sent, SendMessage) and len(sent.content.split()) == 45
    assert log_lines(tmp_path)[0]["trimmed"] is True


def test_a_crash_while_chatting_terminates(tmp_path, capsys):
    assert agent(tmp_path, message=RuntimeError("bad")).choose_message(state(), ctx()) is TERMINATE_MESSAGING
    assert "FALLBACK message" in capsys.readouterr().out
    assert log_lines(tmp_path)[0]["source"] == "fallback"


@pytest.mark.parametrize("message, problem", [
    (SendMessage("   "), "empty"),
    (SendMessage("psst", [2]), "not a living other player"),  # myself
    (SendMessage("psst", [5]), "not a living other player"),  # dead
    (SendMessage("psst", [9]), "not a living other player"),  # no such seat
    (SendMessage("psst", [3, 4]), "one seat"),
    ("hello", "not SendMessage"),
])
def test_a_bad_chat_terminates_instead_of_being_sent(tmp_path, message, problem):
    a = agent(tmp_path, message=message)
    assert a.choose_message(state(alive=[0, 1, 2, 3, 4, 6]), ctx()) is TERMINATE_MESSAGING
    assert problem in log_lines(tmp_path)[0]["error"]


def test_a_private_chat_to_a_living_player_is_kept_private(tmp_path):
    sent = agent(tmp_path, message=SendMessage("kill Player3 tonight", [4])).choose_message(state(), ctx())
    assert sent.recipients == [4]


def test_never_a_sixth_chat_in_one_day(tmp_path):
    a = agent(tmp_path, message=SendMessage("again"))
    sent = [a.choose_message(state(day=1), ctx()) for _ in range(7)]
    assert [isinstance(m, SendMessage) for m in sent] == [True] * 5 + [False] * 2
    assert isinstance(a.choose_message(state(day=2), ctx()), SendMessage)  # a new day starts over


# -- results and notes ------------------------------------------------------------------


def test_move_results_and_werewolf_notes_are_logged(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WEREWOLF_DEBUG", raising=False)
    a = agent(tmp_path, action="4")
    a.choose_action(state(legal=DAY_VOTE), ctx())
    a._note("[werewolf] P2 (villager) day_vote: -> 4")
    a.on_action_result({"accepted": False, "error": "STALE_STATE", "detail": "moved on"}, ctx())
    kinds = [(l["kind"], l.get("text"), l.get("error")) for l in log_lines(tmp_path)]
    assert kinds[1:] == [("note", "[werewolf] P2 (villager) day_vote: -> 4", None), ("action_result", None, "STALE_STATE")]
    assert capsys.readouterr().out == ""  # notes print only with WEREWOLF_DEBUG=1


def test_a_broken_log_never_breaks_the_game(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    a = Agent(delegate=Scripted(action="4"), decision_log=DecisionLog(blocker))
    assert a.choose_action(state(legal=DAY_VOTE), ctx()) == "4"
    assert a.choose_action(state(legal=DAY_VOTE), ctx()) == "4"
    assert capsys.readouterr().out.count("[decision-log] stopped logging") == 1


def test_the_log_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("ALTRUAGENT_DECISION_LOG", "0")
    Agent(delegate=Scripted(action="4"), decision_log=DecisionLog(tmp_path)).choose_action(state(legal=DAY_VOTE), ctx())
    assert not (tmp_path / "w_1").exists()
