"""Cross-game memory, step 1: one record per seat per finished Werewolf game.

Each seat collects what it saw during the game (every chat message, its
role and private info, the latest observation) and, when the runtime reports
the game over (``on_game_end``), appends one JSON line to
``memory/werewolf_games.jsonl`` (gitignored). An offline script turns these
records into lessons; games themselves never read this file.

``ALTRUAGENT_GAME_MEMORY=0`` turns it off; ``ALTRUAGENT_GAME_MEMORY_DIR``
writes somewhere else. Writing never raises: a full disk or a bad path costs
the record, never the game.

Opponents' messages are stored as data for later analysis. Nothing here is
ever fed back to a model verbatim.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from altruagent import DecisionContext, GameState

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "memory"
FILENAME = "werewolf_games.jsonl"
VERSION = 1

try:  # one line per record even when all 7 seats finish at the same moment
    import fcntl
except ImportError:  # Windows: appends of one short line are atomic enough in practice
    fcntl = None


class GameMemory:
    def __init__(self, directory: str | Path | None = None, *, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = os.environ.get("ALTRUAGENT_GAME_MEMORY", "1") != "0"
        self._enabled = enabled
        self._dir = Path(directory or os.environ.get("ALTRUAGENT_GAME_MEMORY_DIR") or DEFAULT_DIR)
        self._chat: list[dict] = []
        self._seen: set[int] = set()
        self._observation: str | None = None
        self._private: dict[str, Any] = {}
        self._teams: dict[int, str] = {}
        self._names: dict[int, str] = {}  # seat -> display name as the server writes it (vote history uses it)
        self._written = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def observe(self, state: GameState, context: DecisionContext) -> None:
        """Call on every decision: keeps every chat message (ours included) and what we know privately."""
        if not self._enabled:
            return
        try:
            raw = state.raw if isinstance(state.raw, dict) else {}
            day = (raw.get("game_state") or {}).get("day")
            for m in state.new_messages:
                if m.index in self._seen or m.type != "chat":
                    continue
                self._seen.add(m.index)
                self._chat.append({"day": day, "seq": m.index, "from": m.sender,
                                   "to": list(m.recipients), "text": m.content})
            self._observation = state.observation
            from agent.werewolf.parsing import seat_teams, snapshot

            self._teams = seat_teams(raw) or self._teams
            for p in raw.get("players") or []:
                if isinstance(p, dict) and isinstance(p.get("position"), int) and p.get("name"):
                    self._names[p["position"]] = str(p["name"])[:120]

            snap, _ = snapshot(state, context.seat_position)
            self._private = {"role": snap.role, "partners": sorted(snap.partners),
                             "seer_results": {str(k): v for k, v in sorted(snap.seer_results.items())}}
        except Exception:  # noqa: BLE001 - memory must never break a game
            pass

    def finish(self, final_state: Any, context: DecisionContext, outcome: str | None, *, agent: str) -> None:
        """Append this seat's record of the game. ``agent`` names who played the seat ("ours", a model...)."""
        if not self._enabled or self._written:
            return
        self._written = True
        try:
            last_day = self._chat[-1]["day"] if self._chat else None
            for m in getattr(final_state, "new_messages", None) or []:  # the closing chat (runner catch-up)
                if m.index not in self._seen and m.type == "chat":
                    self._seen.add(m.index)
                    self._chat.append({"day": last_day, "seq": m.index, "from": m.sender,
                                       "to": list(m.recipients), "text": m.content})
            raw = getattr(final_state, "raw", None)
            record = {
                "v": VERSION,
                "time": round(time.time(), 3),
                "session": context.session_id,
                "kind": "tournament" if context.tournament_id else "test",
                "tournament_id": context.tournament_id,
                "seat": context.seat_position,
                "agent": agent,
                **self._private,
                "outcome": outcome,
                "teams": {str(k): v for k, v in sorted(self._teams.items())},
                "names": {str(k): v for k, v in sorted(self._names.items())},
                "returns": getattr(final_state, "returns", None),
                "observation": getattr(final_state, "observation", None) or self._observation,
                "last_observation": self._observation,
                "game_state": raw.get("game_state") if isinstance(raw, dict) else None,
                "final_result": getattr(final_state, "final_result", None),
                "chat": self._chat,
            }
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._dir / FILENAME, "a", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_EX)
                f.write(line)
                f.flush()
        except Exception as exc:  # noqa: BLE001 - memory must never break a game
            print(f"[game-memory] couldn't save this game: {exc!r}")
