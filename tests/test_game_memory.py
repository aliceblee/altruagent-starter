"""Tests for cross-game memory records (agent/game_memory.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.agent import Agent
from agent.game_memory import FILENAME, GameMemory
from tests.test_werewolf_agent import ctx, state


def _final(**kw):
    return SimpleNamespace(raw={"game_state": {"day": 2, "phase": "ended"}}, returns={"a": 1.0},
                           observation="Game over.", final_result={"status": "completed"}, **kw)


def _records(tmp_path):
    return [json.loads(line) for line in (tmp_path / FILENAME).read_text().splitlines()]


def test_a_seat_saves_one_record_with_its_role_and_the_whole_chat(tmp_path):
    memory = GameMemory(tmp_path, enabled=True)
    seer = "You are Player2. Your role: seer. Night 1: you investigated Player4: wolf."
    memory.observe(state(2, seer, phase="messaging", messages=[(1, "Hi all."), (2, "Any leads?")]), ctx(2))
    memory.observe(state(2, seer, phase="messaging", messages=[(1, "Hi all."), (2, "Any leads?"), (5, "Player3 is odd.")]), ctx(2))
    memory.finish(_final(), ctx(2), "won", agent="ours:rules")
    memory.finish(_final(), ctx(2), "won", agent="ours:rules")  # only once
    (record,) = _records(tmp_path)
    assert (record["seat"], record["agent"], record["outcome"], record["role"]) == (2, "ours:rules", "won", "seer")
    assert record["seer_results"] == {"4": "wolf"} and record["kind"] == "test"
    assert [(m["from"], m["text"]) for m in record["chat"]] == [(1, "Hi all."), (2, "Any leads?"), (5, "Player3 is odd.")]
    assert record["game_state"] == {"day": 2, "phase": "ended"} and record["final_result"] == {"status": "completed"}


def test_seats_append_to_the_same_file(tmp_path):
    for seat in range(3):
        memory = GameMemory(tmp_path, enabled=True)
        memory.observe(state(seat, "Your role: villager.", phase="messaging"), ctx(seat))
        memory.finish(_final(), ctx(seat), "lost", agent="ours:rules")
    assert [r["seat"] for r in _records(tmp_path)] == [0, 1, 2]


def test_memory_can_be_turned_off_and_never_breaks_a_game(tmp_path, monkeypatch):
    monkeypatch.setenv("ALTRUAGENT_GAME_MEMORY", "0")
    GameMemory(tmp_path).finish(_final(), ctx(2), "won", agent="x")
    assert not (tmp_path / FILENAME).exists()
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    GameMemory(blocked, enabled=True).finish(_final(), ctx(2), "won", agent="x")  # prints, doesn't raise


def test_our_agent_records_werewolf_games(tmp_path):
    agent = Agent(brain="rules", memory=GameMemory(tmp_path, enabled=True))
    agent.choose_message(state(6, "Your role: villager.", phase="messaging", messages=[(3, "Hello.")]), ctx(6))
    agent.on_game_end(_final(), ctx(6), "lost")
    (record,) = _records(tmp_path)
    assert record["agent"] == "ours:rules" and record["role"] == "villager"
    assert [m["text"] for m in record["chat"]] == ["Hello."]
