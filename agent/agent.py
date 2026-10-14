"""Your agent's decision logic.

`create_agent()` is called once per game, in that game's own process. It
returns a small router that picks the right player the first time the
runtime asks for a decision:

- **Werewolf**, with `ANTHROPIC_API_KEY` set ->
  `agent.werewolf.claude_brain.ClaudeWerewolfAgent`: Claude makes every
  decision and writes every chat, following the team playbook
  (`agent/werewolf/playbook.md`) and advised by the rule engine below, which
  also plays whenever Claude fails or times out. `WEREWOLF_BRAIN=rules`
  turns Claude off.
- **Werewolf** otherwise -> `agent.werewolf.WerewolfAgent`: Bayesian role
  inference over every possible role assignment, role-specific play (seer /
  wolf / villager), and planned table talk. If `OPENAI_API_KEY` is set, an
  LLM only rewrites the planned chat messages (disable with
  `WEREWOLF_LLM_CHAT=0`). Set `WEREWOLF_DEBUG=1` to log every decision and
  the first private observation.
- **Any other game** -> `examples.llm_agent.LLMAgent` when `OPENAI_API_KEY`
  is set, otherwise the first legal action (which can't finish Pokémon or
  Red Alert; see GAMES.md).

**Werewolf safety net.** The router checks every Werewolf decision before
the runtime sees it, because an error there stops the game's process for
about a minute while the server plays for us. If `WerewolfAgent` raises or
returns a move that isn't legal, the router plays what the server would on
a timeout (abstain by day, the first legal target at night). If a chat is
too long it is trimmed; if it is empty, a 6th chat of the day, or addressed
to a seat that isn't a living other player, the router terminates instead
(never broadcasting a message meant to be private). Each fallback prints a
`[werewolf] FALLBACK` line.

**Decision log.** Every Werewolf decision, chat, fallback and move result is
appended to `logs/decisions/<session_id>/seat<N>.jsonl` (see
`agent/decision_log.py`; `ALTRUAGENT_DECISION_LOG=0` turns it off).

Try the Werewolf agent offline against baselines before a tournament:

    python -m sim.werewolf_sim --games 500
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any

from dotenv import load_dotenv

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState, LegalAction, SendMessage, WithReasoning
from agent.decision_log import DecisionLog

ABSTAIN = "7"
MAX_CHATS_PER_DAY = 5  # the server's limit
MAX_WORDS = 45  # the server's cap is 50
MAX_CHARS = 1000  # the server's cap is 1024


class Agent:
    def __init__(self, *, delegate: Any = None, decision_log: DecisionLog | None = None,
                 brain: str | None = None) -> None:
        self._delegate: Any = delegate
        self._brain = brain  # "claude" / "rules"; None: WEREWOLF_BRAIN, else Claude when ANTHROPIC_API_KEY is set
        self._decisions = decision_log or DecisionLog()
        self._context: DecisionContext | None = None
        self._chats_by_day: dict[Any, int] = {}
        self._heard: set[int] = set()

    def _player(self, context: DecisionContext) -> Any:
        if self._delegate is None:
            if _is_werewolf(context):
                from agent.werewolf import WerewolfAgent
                from agent.werewolf.chat_llm import polisher_from_env
                from agent.werewolf.claude_brain import claude_agent_from_env

                if self._brain != "rules":
                    self._delegate = claude_agent_from_env(log=self._note, brain=self._brain)
                if self._delegate is None:
                    self._delegate = WerewolfAgent(polish=polisher_from_env(), log=self._note)
            elif os.environ.get("OPENAI_API_KEY"):
                from examples.llm_agent import create_agent as create_llm_agent

                self._delegate = create_llm_agent()
            else:
                self._delegate = _FirstLegal()
        return self._delegate

    def choose_action(self, state: GameState, context: DecisionContext):
        player = self._player(context)
        if not _is_werewolf(context):
            return player.choose_action(state, context)
        self._context = context
        started = time.monotonic()
        error = None
        try:
            decision = player.choose_action(state, context)
            if _legal_action(decision, state) is None:
                error = f"returned {decision!r}, which is not a legal action"
        except Exception as exc:  # noqa: BLE001 - the safety net
            error = _describe(exc)
        if error is not None:
            decision = _fallback_action(state)
            print(f"[werewolf] FALLBACK move ({_sub_phase(state)}): {error}")
        action = _legal_action(decision, state)
        self._decisions.write(
            context, "action", day=_day(state), sub_phase=_sub_phase(state),
            legal=[a.action_id for a in state.legal_actions], action_id=action.action_id, label=action.label,
            source="fallback" if error else "agent", error=error, ms=_ms(started), observation=state.observation,
        )
        return decision

    def choose_message(self, state: GameState, context: DecisionContext):
        method = getattr(self._player(context), "choose_message", None)
        if method is None:
            return TERMINATE_MESSAGING
        if not _is_werewolf(context):
            return method(state, context)
        self._context = context
        started = time.monotonic()
        day = _day(state)
        for heard in state.new_messages:
            if heard.index not in self._heard and heard.type == "chat":
                self._heard.add(heard.index)
                self._decisions.write(context, "heard", day=day, seq=heard.index, sender=heard.sender,
                                      recipients=list(heard.recipients), content=heard.content)
        trimmed = False
        try:
            decision, problem, trimmed = _checked_message(method(state, context), state, self._chats_by_day.get(day, 0))
        except Exception as exc:  # noqa: BLE001 - the safety net
            decision, problem = None, _describe(exc)
        if problem is not None:
            decision = TERMINATE_MESSAGING
            print(f"[werewolf] FALLBACK message (terminating instead): {problem}")
        if isinstance(decision, SendMessage):
            self._chats_by_day[day] = self._chats_by_day.get(day, 0) + 1
        self._decisions.write(
            context, "message", day=day, seconds_left=state.raw.get("messaging_seconds_left"),
            messages_seen=len(state.new_messages),
            type="chat" if isinstance(decision, SendMessage) else "terminate",
            content=getattr(decision, "content", None), recipients=getattr(decision, "recipients", None),
            trimmed=trimmed, source="fallback" if problem else "agent", error=problem, ms=_ms(started),
        )
        return decision

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        if _is_werewolf(context):
            self._decisions.write(
                context, "action_result", accepted=result.get("accepted"),
                error=result.get("error"), detail=result.get("detail"),
            )
        method = getattr(self._delegate, "on_action_result", None)
        if method is not None:
            method(result, context)

    def _note(self, message: str) -> None:
        """WerewolfAgent's own log lines: printed with WEREWOLF_DEBUG=1, always kept in the decision log."""
        if os.environ.get("WEREWOLF_DEBUG"):
            print(message)
        if self._context is not None:
            self._decisions.write(self._context, "note", text=message)


class _FirstLegal:
    def choose_action(self, state: GameState, context: DecisionContext):
        return state.legal_actions[0]


def _is_werewolf(context: DecisionContext) -> bool:
    return (context.game_type or "").startswith("werewolf")


def _legal_action(decision: Any, state: GameState) -> LegalAction | None:
    """The legal action a decision names (a LegalAction, its action_id, a
    matching int, or any of those wrapped in WithReasoning), or None."""
    if isinstance(decision, WithReasoning):
        decision = decision.action
    if isinstance(decision, LegalAction):
        decision = decision.action_id
    if isinstance(decision, int) and not isinstance(decision, bool):
        decision = str(decision)
    return next((a for a in state.legal_actions if a.action_id == decision), None) if isinstance(decision, str) else None


def _fallback_action(state: GameState) -> LegalAction:
    """What the server plays on a timeout: abstain by day, the first legal target at night."""
    return next((a for a in state.legal_actions if a.action_id == ABSTAIN), state.legal_actions[0])


def _checked_message(decision: Any, state: GameState, chats_today: int) -> tuple[Any, str | None, bool]:
    """(message to send, problem or None, whether it was trimmed)."""
    if decision is TERMINATE_MESSAGING:
        return decision, None, False
    if not isinstance(decision, SendMessage):
        return None, f"returned {decision!r}, not SendMessage or TERMINATE_MESSAGING", False
    if chats_today >= MAX_CHATS_PER_DAY:
        return None, f"would be chat number {chats_today + 1} today; the limit is {MAX_CHATS_PER_DAY}", False
    words = str(decision.content or "").split()
    text = " ".join(words[:MAX_WORDS])[:MAX_CHARS]
    if not text:
        return None, "the message is empty", False
    recipients = decision.recipients
    if len(recipients) > 1:
        return None, f"a private message goes to one seat, not {recipients}", False
    if recipients:
        recipient = recipients[0]
        if isinstance(recipient, bool) or recipient not in _living_others(state):
            return None, f"recipient {recipient!r} is not a living other player", False
    trimmed = text != " ".join(words)
    return (SendMessage(text, recipients) if trimmed else decision), None, trimmed


def _living_others(state: GameState) -> set[int]:
    from agent.werewolf.parsing import snapshot

    snap, _ = snapshot(state)
    return set(snap.alive) - {snap.me}


def _day(state: GameState) -> Any:
    return (state.raw.get("game_state") or {}).get("day")


def _sub_phase(state: GameState) -> Any:
    return (state.raw.get("game_state") or {}).get("phase")


def _describe(exc: Exception) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    where = f" at {os.path.basename(frames[-1].filename)}:{frames[-1].lineno}" if frames else ""
    return f"{type(exc).__name__}: {exc}{where}"


def _ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def create_agent() -> Agent:
    load_dotenv()  # no-op if already loaded; never overrides real env vars
    return Agent()
