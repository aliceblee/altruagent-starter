"""Phase 0 recorder: plays Werewolf trivially and writes down everything it sees.

    python -m agent --match --agent agent.recorder

Meant for a self-play test match (all 7 seats yours). Each seat appends one
JSON line per call (choose_action, choose_message, on_action_result) to
``recordings/<session_id>/seat<N>.jsonl``, holding the full ``state.raw``,
so we can learn the real payloads before writing the real agent.

Moves: the first legal action (nights and votes alike), so the game ends.

Messaging runs a few scripted probes on Day 1 only (every seat is alive
then, so every private recipient is valid):

- every seat: one broadcast, one private message to the next seat, then
  terminate;
- seat 1 holds the window open: it keeps chatting (up to 4 chats) before
  terminating, so the others get called again after they terminated;
- seat 0, on its first call after terminating, sends one more chat. This is
  the question "is a chat after terminating accepted?". If the server
  refuses it, seat 0's process stops with a DecisionError and comes back
  about a minute later; the record line written before the send, plus the
  console, says what happened.

Later days: one broadcast, then terminate.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState, SendMessage

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"
HOLDER_SEAT = 1  # keeps the Day 1 window open
PROBE_SEAT = 0  # chats once after terminating, on Day 1
HOLDER_CHATS = 4


class RecorderAgent:
    def __init__(self) -> None:
        self._file = None
        self._day_calls: dict[object, int] = {}  # messaging calls per day
        self._terminated_days: set[object] = set()
        self._probe_sent = False

    # -- moves -------------------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        action = state.legal_actions[0]
        self._record("choose_action", state, context, decision={"action_id": action.action_id, "label": action.label})
        return action

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        self._record("action_result", None, context, decision=result)

    # -- messaging -------------------------------------------------------------------

    def choose_message(self, state: GameState, context: DecisionContext):
        day = (state.raw.get("game_state") or {}).get("day")
        call = self._day_calls.get(day, 0) + 1
        self._day_calls[day] = call
        decision = self._message_plan(state, context, day, call)

        if decision is TERMINATE_MESSAGING:
            self._terminated_days.add(day)
            summary = {"type": "terminate"}
        else:
            summary = {"type": "chat", "content": decision.content, "recipients": decision.recipients}
        self._record("choose_message", state, context, decision=summary, day=day, call=call)
        return decision

    def _message_plan(self, state: GameState, context: DecisionContext, day, call: int):
        seat = _seat(state, context)
        tag = f"[rec seat{seat} day{day} call{call}]"
        if day != 1:
            return SendMessage(f"{tag} broadcast") if call == 1 else TERMINATE_MESSAGING

        if day in self._terminated_days:
            if seat == PROBE_SEAT and not self._probe_sent:
                self._probe_sent = True
                return SendMessage(f"{tag} chat after terminate (probe)")
            return TERMINATE_MESSAGING

        if call == 1:
            return SendMessage(f"{tag} broadcast")
        if call == 2:
            return SendMessage(f"{tag} private to seat{(seat + 1) % 7}", [(seat + 1) % 7])
        if seat == HOLDER_SEAT and call <= HOLDER_CHATS:
            return SendMessage(f"{tag} holding the window open")
        return TERMINATE_MESSAGING

    # -- recording -----------------------------------------------------------------

    def _record(self, kind: str, state: GameState | None, context: DecisionContext, **extra) -> None:
        line = {
            "kind": kind,
            "wall_time": time.time(),
            "seat": _seat(state, context),
            "game_type": context.game_type,
            **extra,
        }
        if state is not None:
            line.update(
                phase=state.phase,
                state_version=state.state_version,
                is_current_actor=state.is_current_actor,
                observation=state.observation,
                legal_actions=[{"action_id": a.action_id, "label": a.label} for a in state.legal_actions],
                new_messages=[vars(m) for m in state.new_messages],
                terminated_messaging=state.terminated_messaging,
                raw=state.raw,
            )
        self._open(context, line["seat"]).write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def _open(self, context: DecisionContext, seat):
        if self._file is None:
            session = re.sub(r"[^A-Za-z0-9_.-]", "_", str(context.session_id))
            folder = RECORDINGS_DIR / session
            folder.mkdir(parents=True, exist_ok=True)
            self._file = open(folder / f"seat{seat}.jsonl", "a", encoding="utf-8")
        return self._file


def _seat(state: GameState | None, context: DecisionContext):
    if context.seat_position is not None:
        return context.seat_position
    return state.raw.get("your_position") if state is not None else "unknown"


def create_agent() -> RecorderAgent:
    return RecorderAgent()
