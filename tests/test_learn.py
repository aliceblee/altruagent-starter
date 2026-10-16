"""Tests for the cross-game learning step (agent/werewolf/learn.py)."""

from __future__ import annotations

import json

from agent.werewolf import learn


def _record(session, seat, agent, roles, chat, votes):
    names = {s: f"alime588 (Player{s}, Self-hosted)" for s in range(7)}
    return {
        "session": session, "seat": seat, "agent": agent, "role": roles[seat], "chat": chat,
        "game_state": {"dead": [], "vote_history": [
            {"day": d, "votes": {names[v]: (names[t] if t is not None else None) for v, t in vs.items()}}
            for d, vs in votes.items()]},
        "final_result": {"result": {"seats": {f"id{s}": {"position": s, "role": r} for s, r in roles.items()}}},
    }


def _game(session, wolf_counterclaims):
    roles = {0: "villager", 1: "wolf", 2: "wolf", 3: "seer", 4: "villager", 5: "villager", 6: "villager"}
    chat = [{"day": 1, "seq": 0, "from": 3, "to": [], "text": "I am the Seer. I checked Player1: WOLF."}]
    if wolf_counterclaims:
        chat.append({"day": 1, "seq": 1, "from": 1, "to": [], "text": "I’m the Seer, and Player3 is the wolf."})
    votes = {1: {0: 1, 1: 3, 2: 3, 3: 1, 4: 1, 5: 1, 6: None}}
    return [_record(session, s, "ours:claude" if s == 0 else "model", roles, chat, votes) for s in range(7)]


def test_learns_opponent_patterns_from_memory_records(tmp_path):
    path = tmp_path / "games.jsonl"
    lines = [r for i in range(5) for r in _game(f"g{i}", wolf_counterclaims=True)]
    path.write_text("\n".join(json.dumps(r) for r in lines))
    games = learn.load_memory(path)
    assert len(games) == 5 and games[0].roles[1] == "wolf" and games[0].votes[1][6] is None
    tags = learn.features(games[0])
    assert {"claimed_seer", "counterclaimed", "accused_claimant", "voted_claimant"} <= tags[1]
    assert "abstained" in tags[6] and "claimed_seer_day1" in tags[3]
    counts = learn.tally(games)
    cc = counts["opponents"]["counterclaimed"]
    assert (cc.wolf, cc.village) == (5, 0) and counts["ours"]["counterclaimed"].seen == 0  # seat 0 is ours
    text = learn.learned_text(counts["opponents"], len(games))
    assert "claimed seer after someone else already had was a wolf 5 of 5 times" in text
    assert "I’m the Seer" not in text  # counts only, never opponents' words


def test_patterns_with_too_few_examples_are_left_out(tmp_path):
    path = tmp_path / "games.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _game("g", wolf_counterclaims=True)))
    counts = learn.tally(learn.load_memory(path))
    assert "claimed seer after someone else" not in learn.learned_text(counts["opponents"], 1)


def test_the_model_learns_from_tournament_games_only_and_starts_fresh(tmp_path):
    test_games = [r for i in range(5) for r in _game(f"t{i}", wolf_counterclaims=True)]
    real = [dict(r, kind="tournament") for i in range(learn.TOURNAMENT_MIN_GAMES) for r in _game(f"r{i}", True)]
    path = tmp_path / "games.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in test_games + real[:7]))  # 5 test games + 1 tournament game
    files = dict(memory_path=path, logs=tmp_path / "no-logs")
    picked, source = learn.opponent_games(learn.load_memory(path))
    assert picked == [] and "1 tournament games (need" in source
    learned, report = learn.build(**files)
    assert learned == "" and "# Werewolf learning report: 6 games" in report  # the report still covers all
    path.write_text("\n".join(json.dumps(r) for r in test_games + real))
    picked, _ = learn.opponent_games(learn.load_memory(path))
    assert {g.kind for g in picked} == {"tournament"} and len(picked) == learn.TOURNAMENT_MIN_GAMES
    learned, _ = learn.build(**files)
    assert f"Learned from {learn.TOURNAMENT_MIN_GAMES} tournament games" in learned


def test_refresh_writes_both_files_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("WEREWOLF_AUTO_LEARN", "1")
    path = tmp_path / "games.jsonl"
    path.write_text("\n".join(json.dumps(dict(r, kind="tournament"))
                               for i in range(learn.TOURNAMENT_MIN_GAMES) for r in _game(f"g{i}", True)))
    files = dict(memory_path=path, logs=tmp_path / "no-logs", learned_path=tmp_path / "learned.md",
                 report_path=tmp_path / "report.md", lock_path=tmp_path / ".lock")
    assert learn.refresh(**files) is True
    assert f"wolf {learn.TOURNAMENT_MIN_GAMES} of {learn.TOURNAMENT_MIN_GAMES} times" in (tmp_path / "learned.md").read_text()
    assert (tmp_path / "report.md").read_text().startswith("# Werewolf learning report")
    monkeypatch.setattr(learn, "build", lambda *a: 1 / 0)
    assert learn.refresh(**files) is False  # a broken build costs the refresh, not the game
    monkeypatch.setenv("WEREWOLF_AUTO_LEARN", "0")
    assert learn.refresh(**files) is False
