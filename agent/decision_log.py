"""Per-match decision log: one JSON line per decision, for tuning later.

Each game's process appends to ``logs/decisions/<session_id>/seat<N>.jsonl``
(gitignored). ``ALTRUAGENT_DECISION_LOG=0`` turns it off;
``ALTRUAGENT_DECISION_LOG_DIR`` writes somewhere else. Writing never raises:
a full disk or a bad path costs the log, never the game.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from altruagent import DecisionContext

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "logs" / "decisions"


class DecisionLog:
    def __init__(self, directory: str | Path | None = None, *, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = os.environ.get("ALTRUAGENT_DECISION_LOG", "1") != "0"
        self._enabled = enabled
        self._dir = Path(directory or os.environ.get("ALTRUAGENT_DECISION_LOG_DIR") or DEFAULT_DIR)
        self._file = None

    def write(self, context: DecisionContext, kind: str, **fields) -> None:
        if not self._enabled:
            return
        try:
            if self._file is None:
                session = re.sub(r"[^A-Za-z0-9_.-]", "_", str(context.session_id))
                folder = self._dir / session
                folder.mkdir(parents=True, exist_ok=True)
                self._file = open(folder / f"seat{context.seat_position}.jsonl", "a", encoding="utf-8")
            line = {"time": round(time.time(), 3), "kind": kind, "seat": context.seat_position, **fields}
            self._file.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            self._file.flush()
        except Exception as exc:  # noqa: BLE001 - logging must never break a game
            self._enabled = False
            print(f"[decision-log] stopped logging for this match: {exc!r}")
