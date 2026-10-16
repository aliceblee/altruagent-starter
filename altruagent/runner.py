"""Single-match execution primitive — MCP-first.

Owns exactly one match's play loop: fetch state, decide (via contestant-
supplied logic) whether a decision is needed right now, submit it, repeat
until the match ends. This is the ONLY production gameplay path — it plays
through ``MCPGameSession`` (``altruagent.mcp_game``), Agent_ACP's generic
MCP gameplay contract, never the lower-level REST ``GameSession``
(``altruagent.game``, kept only for ``scripts/check_game.py``'s manual
debugging). This file contains no per-game branching (no ``if game_type ==
"pokemon"``/``"avalon"``/...) — every game (OpenSpiel-family or a structured
RuntimeAdapter like Pokémon) is driven through the same four generic
observations: ``is_terminal``, ``phase`` (only the literal value
``"messaging"`` is special-cased), ``is_current_actor``, and
``legal_actions`` — confirmed, not assumed, to be returned under those exact
key names by every adapter currently in Agent_ACP
(``gameapi/src/gameapi/runtime_adapters/{openspiel_adapter,pokemon_adapter}.py``).

The contestant contract is two parameters and a return value, nothing more:

    def choose_action(state: GameState, context: DecisionContext) -> int:
        return state.legal_actions[0]

No base class, no decorator, no registration. `run_match`/`run_game` also
accept an object exposing a `.choose_action(state, context)` method instead
of a plain function, resolved with one `callable()` check and one
`getattr()` — no signature/arity introspection.

`choose_action` may return, and this runner normalizes without ever
guessing or fuzzy-coercing:

    - a `LegalAction` from `state.legal_actions` (`return
      state.legal_actions[0]` is enough for Werewolf and the Pokémon draft,
      but not for Pokémon's Team Preview/doubles templates or Red Alert,
      whose `state.legal_actions` is empty; see GAMES.md)
    - that `LegalAction`'s `action_id` string
    - a plain `int`, ONLY accepted when `str(that int)` exactly equals some
      current legal action's `action_id` (this is what makes OpenSpiel-family
      games' historical `return 0`-style agents keep working unchanged — and
      why it correctly REJECTS an int for a structured game like Pokémon,
      whose action_ids are `"move:0"`, not `"0"`)
    - a structured `dict`, passed straight through as the MCP `action`
      payload for constructive actions (e.g. Pokémon's `submit_team`) that
      can't be enumerated as one of `state.legal_actions` — the SDK performs
      no game-specific validation of it; the server is authoritative
    - `altruagent.RESIGN`
    - `altruagent.WAIT`, only in a real-time game (see below)
    - `altruagent.WithReasoning(<any of the above except RESIGN>, "...")` —
      the same move plus a short public `reasoning_summary`, sent through
      MCP `play_action`'s own optional argument of that name

Turn detection is driven by `GameState.is_current_actor`/`phase`/
`is_terminal` (plus Werewolf's `eliminated` flag) rather than MCP's
`next_actions` hint, so the loop stays independent of hint wording.
`next_actions` is still parsed onto `GameState` but is supplementary/
informational only.

Waiting is a server-side long-poll: when it isn't this agent's turn, the
runner calls `wait_for_update`, which returns as soon as anything changes
(a move, a new message, whose turn it is, the phase, or the game ending).
Each wait lasts at most `WAIT_FOR_UPDATE_TIMEOUT_SECONDS` (then it simply
waits again); against a server that predates that tool, the runner sleeps
`wait_seconds` and re-reads state instead.

Messaging works the same way: `phase == "messaging"` is the one phase value
this runner recognizes by exact string match; every other phase value (a
future adapter's `"moving"`, `"draft"`, `"teambuild"`, or anything else)
falls through to the identical "it's my turn -> enumerate legal actions ->
choose_action" path with zero adapter-specific code — this is what lets
Pokémon's draft/teambuild phases work without this runner knowing Pokémon
exists. `choose_message` is optional — a contestant that only defines
`choose_action` gets `TERMINATE_MESSAGING` automatically every round (see
`_default_choose_message`), so Werewolf (whose discussion window opens once
per day) can be played move-only with zero new contestant code. `choose_message` is never invoked for a game whose adapter
never reports `phase == "messaging"` (confirmed: Pokémon's phases are
`draft`/`draft_complete`/`teambuild`/`moving`, never `"messaging"`).

Real-time games (a state whose `pacing.mode` is `"realtime"`, e.g. Red
Alert) have no turns: both players are "current" at once and the world keeps
moving while an agent thinks. The loop is the same, with three differences,
all keyed on that pacing field rather than on a game name:

    - `choose_action` may return `altruagent.WAIT`: nothing to send right
      now; the runner waits for the next state (`wait_for_update`, which in
      real time returns as soon as a newer view exists) and asks again.
    - A move the server refuses as a whole (`INVALID_ACTION`, e.g. every
      unit in an order died between the read and the send) is a race, not a
      contestant bug: the runner re-reads the state and continues instead of
      raising `DecisionError`. `RUNTIME_TEMPORARILY_UNAVAILABLE` is retried
      after `REALTIME_RETRY_SECONDS`.
    - Before the first decision the runner fetches the game's reference
      (`get_game_config`) once, best-effort, and hands it to decision logic
      as `context.game_config`.

An agent object may also define `on_action_result(result, context)`: the
runner calls it after every `play_action` with the server's answer, or with
`{"error": <code>, "detail": <message>}` when the server refused the move
and the runner recovered. That is how an agent learns that a move it sent
was refused, for any game. It is optional; a plain function agent never
needs it.

`state_version` (MCP's optimistic-concurrency counter) is entirely
runner-owned: taken from the same read that produced `state.legal_actions`
(the server only embeds legal actions whose `state_version` matches the
state's), never something `choose_action`/`choose_message` supply or manage.
"""

from __future__ import annotations

import linecache
import os
import random
import re
import sysconfig
import time
import traceback
from dataclasses import replace
from typing import Any, Callable, NamedTuple, Protocol, Union

from .errors import AltruAgentError, is_transient_error
from .mcp_game import MCPGameSession
from .mcp_transport import MCPToolError
from .models import DecisionContext, GameState, LegalAction, Match, Message

# The sleep between reads against a server without wait_for_update.
DEFAULT_WAIT_SECONDS = 5.0

# A temporary failure of any gameplay call (a dropped connection, a gateway or
# server error, a game engine that is busy: ``errors.is_transient_error``)
# doesn't end the game. The runner pauses, then reads the state again and
# carries on; a move or message is never sent again blindly (it may have
# landed), the decision is simply made again from the fresh state. The pause
# grows from about 1 s to 5 s, with jitter. Only after this long without one
# successful call does the error reach the worker.
TRANSIENT_RETRY_FIRST_SECONDS = 1.0
TRANSIENT_RETRY_MAX_SECONDS = 5.0
TRANSIENT_GIVE_UP_SECONDS = 90.0

# GameAPI lets only a couple of wait_for_update calls wait at once per agent
# and game; another answers TOO_MANY_WAITS. A call this runtime already gave
# up on (a dropped connection) can still be waiting there for up to 25 s, so
# this is paced and retried like a temporary failure, never the agent's fault.
_WAIT_BUSY_ERROR_CODE = "TOO_MANY_WAITS"

# wait_for_update's long-poll timeout; it returns earlier as soon as anything
# changes. 20 s is the server default (its maximum is 25 s), well under the
# MCP client's 40 s read timeout (``mcp_transport.MCP_READ_TIMEOUT_SECONDS``).
# This relies on the server honoring since_is_current_actor/
# since_phase (Agent_ACP #24): without them, a whose-turn/phase change that
# lands before the call — every Pokémon battle turn for the second seat to
# submit — is only noticed at the timeout.
WAIT_FOR_UPDATE_TIMEOUT_SECONDS = 20.0

# Confirmed exact codes against Agent_ACP's gameapi/src/gameapi/mcp_server/errors.py
# (`_DOMAIN_ERROR_CODES`, plus the documented fallback of an unlisted domain
# exception's own `.error` attribute uppercased, e.g. WrongPhaseError's
# "wrong_phase" -> "WRONG_PHASE"). All four represent "the match state moved
# since the last read" — a genuine race, never a contestant bug — so all four
# just refetch state and re-enter the decision loop from the top.
# PLAYER_ELIMINATED (Werewolf) belongs here too: this agent was eliminated
# after the read it acted on; the refetched state carries `eliminated`, and
# the loop then only waits for the game to end.
_RACE_ERROR_CODES = frozenset(
    {
        "STALE_STATE",
        "NOT_YOUR_TURN",
        "WRONG_PHASE",
        "GAME_ALREADY_COMPLETE",
        "PLAYER_ELIMINATED",
    }
)

# Contestant-caused messaging failures (bad content/recipients, over quota,
# or messaging attempted on a non-messaging game) — fail-fast DecisionErrors,
# same philosophy as an illegal choose_action result.
_MESSAGE_SCOPED_ERROR_CODES = frozenset(
    {
        "INVALID_RECIPIENTS",
        "MESSAGE_TOO_LONG",
        "TOO_MANY_WORDS",
        "INVALID_CONTENT",
        "MESSAGES_QUOTA_EXCEEDED",
        "MESSAGING_DISABLED",
    }
)

# Contestant-caused action failures.
_ACTION_SCOPED_ERROR_CODES = frozenset({"INVALID_ACTION"})

# Real-time games only (``pacing.mode == "realtime"``): the server could not
# deliver the move right now (nothing was applied). Retried after a short pause.
_REALTIME_TRANSIENT_ERROR_CODES = frozenset({"RUNTIME_TEMPORARILY_UNAVAILABLE"})
REALTIME_RETRY_SECONDS = 1.0
_REALTIME_PACING = "realtime"

# The runner believed messaging/a capability was available (phase said so)
# but the adapter disagrees — a genuine runner/adapter mismatch, not a
# contestant bug and not a race; never silently retried.
_CAPABILITY_ERROR_CODE = "RUNTIME_UNAVAILABLE"

_MESSAGING_PHASE = "messaging"


class _Resign:
    """Unique sentinel type. Never instantiate another one — always use the
    exported ``RESIGN`` singleton and compare with ``is``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "RESIGN"


RESIGN = _Resign()


class _Wait:
    """Unique sentinel type, analogous to ``_Resign``. Never instantiate
    another one — always use the exported ``WAIT`` singleton and compare
    with ``is``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "WAIT"


# Returned from ``choose_action`` in a real-time game: nothing to send right
# now. The runner waits for the next state and asks again. A turn-based game
# has no such option (the turn waits for you), so there it's a DecisionError.
WAIT = _Wait()


class _TerminateMessaging:
    """Unique sentinel type, analogous to ``_Resign``. Never instantiate
    another one — always use the exported ``TERMINATE_MESSAGING`` singleton
    and compare with ``is``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "TERMINATE_MESSAGING"


TERMINATE_MESSAGING = _TerminateMessaging()


class SendMessage:
    """Returned from an optional ``choose_message`` to send a chat message
    during a MESSAGING phase.

    ``recipients`` empty/``None`` broadcasts to every other player; a single
    player index sends a targeted p2p message. The server currently rejects
    2+ recipients (p2group is gated) — that surfaces as ``DecisionError``,
    same as any other contestant-caused messaging error (see
    ``_MESSAGE_SCOPED_ERROR_CODES``).
    """

    __slots__ = ("content", "recipients")

    def __init__(self, content: str, recipients: list[int] | None = None) -> None:
        self.content = content
        self.recipients = list(recipients) if recipients else []

    def __repr__(self) -> str:
        return f"SendMessage(content={self.content!r}, recipients={self.recipients!r})"


class WithReasoning:
    """Returned from ``choose_action`` to attach a short PUBLIC explanation to
    a move: ``return WithReasoning(state.legal_actions[0], "Fake Out to stall.")``.

    ``action`` is anything ``choose_action`` may return on its own except
    ``RESIGN``. ``reasoning_summary`` is sent as MCP ``play_action``'s own
    optional ``reasoning_summary`` argument, which GameAPI records alongside
    the move (e.g. a Pokémon draft pick's ``public_reason``) for spectators
    — so write it for the audience; never put secrets or hidden
    deliberation in it.
    """

    __slots__ = ("action", "reasoning_summary")

    def __init__(self, action: Any, reasoning_summary: str | None) -> None:
        self.action = action
        text = reasoning_summary.strip() if isinstance(reasoning_summary, str) else ""
        self.reasoning_summary = text or None

    def __repr__(self) -> str:
        return f"WithReasoning(action={self.action!r}, reasoning_summary={self.reasoning_summary!r})"


Decision = Union[int, str, LegalAction, dict, _Resign, _Wait, WithReasoning]
MessageDecision = Union[SendMessage, _TerminateMessaging]


class ChoosesAction(Protocol):
    """Structural type for an object-based decision handler: anything with a
    ``choose_action(state, context)`` method (and, optionally, a
    ``choose_message(state, context)`` method — see ``MessageDecisionFn``).
    Not used for runtime isinstance checks (see ``_resolve_decision_fn``) —
    this exists only to give type checkers something to check plain-function
    agents against.
    """

    def choose_action(self, state: GameState, context: DecisionContext) -> Decision: ...


DecisionFn = Callable[[GameState, DecisionContext], Decision]
MessageDecisionFn = Callable[[GameState, DecisionContext], MessageDecision]


class RunnerError(Exception):
    """Base class for all altruagent.runner errors."""


class DecisionError(RunnerError):
    """The contestant decision function misbehaved: it raised, returned a
    type/value this runner doesn't recognize (see the module docstring for
    exactly what's accepted), or produced an action/message the server
    rejected as invalid (bad recipients/content, word/length cap, chat quota
    exceeded, malformed structured action — see ``_MESSAGE_SCOPED_ERROR_CODES``/
    ``_ACTION_SCOPED_ERROR_CODES``). Fails fast, on purpose: the worker exits
    so a deterministic contestant bug is visible immediately; the supervisor
    retries the seat later with a growing pause.
    """


class UnsupportedGameFlowError(RunnerError):
    """The match reported a phase/capability this runner expected the
    adapter to support, but the adapter disagreed (MCP's
    ``RUNTIME_UNAVAILABLE``) — a genuine adapter/runner mismatch, not a
    contestant bug. Not raised merely because ``choose_message`` is absent
    (see ``_default_choose_message``) — only when the runner's own
    phase-based detection turns out to be wrong for a given adapter.
    """


class _PlayAction(NamedTuple):
    """Normalized form of a validated ``choose_action`` decision — exactly
    one of ``action_id``/``action`` is set, matching MCP's own
    ``play_action`` precedence (``action`` wins if both are given).
    """

    action_id: str | None
    action: dict | None
    reasoning_summary: str | None = None


def _resolve_decision_fn(choose_action: Any) -> DecisionFn:
    """Accept either a plain callable, or an object exposing a callable
    ``choose_action`` attribute. Two simple checks, no reflection.
    """
    if callable(choose_action):
        return choose_action
    method = getattr(choose_action, "choose_action", None)
    if callable(method):
        return method
    raise DecisionError(
        f"{choose_action!r} is not callable and has no callable 'choose_action' "
        "method. Expected a function like choose_action(state, context), or an "
        "object exposing one."
    )


def _resolve_message_decision_fn(choose_action: Any) -> MessageDecisionFn:
    """Look up an optional ``choose_message(state, context)`` on the same
    object/function used for ``choose_action``.

    Unlike ``_resolve_decision_fn``, a missing ``choose_message`` is not an
    error — it's the normal case for a contestant that only cares about
    moves, and defaults to always terminating the messaging round
    immediately (see ``_default_choose_message``). Must be resolved from the
    *original* ``choose_action`` value the caller passed in, not from
    ``_resolve_decision_fn``'s result — for an object-based agent, that
    result is a bound ``choose_action`` method, and a bound method does not
    proxy attribute lookups back to the object it came from.
    """
    method = getattr(choose_action, "choose_message", None)
    if callable(method):
        return method
    return _default_choose_message


def _default_choose_message(state: GameState, context: DecisionContext) -> MessageDecision:
    return TERMINATE_MESSAGING


# Display names for the "can't play" message below, by game type. Wording
# only: the runner never plays a game differently because of its name.
_GAME_NAMES = {"red_alert": "Red Alert"}


def _cant_play_realtime_message(context: DecisionContext) -> str:
    """What an agent that picks from ``state.legal_actions`` (like the
    placeholder ``agent/agent.py``) is told in a real-time game, where that
    list is empty: plainly that it can't play the game, and where to look,
    instead of a raw ``IndexError``."""
    game_type = context.game_type or ""
    name = _GAME_NAMES.get(game_type) or (f"this real-time game ({game_type})" if game_type else "this real-time game")
    return (
        f"Your agent can't play {name}: it picks its move from state.legal_actions, which is empty "
        "in a real-time game (a move there is a batch of orders your agent writes itself). The "
        "placeholder agent/agent.py works this way. See GAMES.md, or run with --agent "
        "examples.llm_agent, which can play it."
    )


# The standard library's own folders, for ``_picked_from_legal_actions``.
_STDLIB_DIRS = tuple(
    {
        os.path.realpath(path) + os.sep
        for path in (sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib"))
        if path
    }
)
_NAMES_LEGAL_ACTIONS = re.compile(r"\blegal_actions\b")


def _is_stdlib_file(filename: str) -> bool:
    if filename.startswith("<frozen "):
        return True
    path = os.path.realpath(filename)
    return path.startswith(_STDLIB_DIRS) and "site-packages" not in path and "dist-packages" not in path


def _failing_expression(frame: traceback.FrameSummary) -> str:
    """The source of the expression a frame was running when the error was
    raised (Python 3.11+ records its columns, as UTF-8 byte offsets), or the
    whole line when the columns aren't known; "" without the source."""
    line = linecache.getline(frame.filename, frame.lineno or 0)
    if line and frame.end_lineno == frame.lineno and frame.colno is not None and frame.end_colno is not None:
        return line.encode("utf-8")[frame.colno : frame.end_colno].decode("utf-8", "replace")
    return line


def _picked_from_legal_actions(exc: BaseException) -> bool:
    """Whether this error came from picking a move out of ``legal_actions``:
    in the innermost frame outside the standard library (so
    ``random.choice(state.legal_actions)`` counts as well as
    ``state.legal_actions[0]``), the expression that failed names
    ``legal_actions``. An agent's own bug, such as ``units[0]`` on an empty
    list, doesn't, and keeps its real error."""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if not _is_stdlib_file(frame.filename):
            return bool(_NAMES_LEGAL_ACTIONS.search(_failing_expression(frame)))
    return False


def _invoke_decision(decision_fn: DecisionFn, state: GameState, context: DecisionContext) -> Any:
    try:
        return decision_fn(state, context)
    except Exception as exc:
        if (
            isinstance(exc, IndexError)
            and not state.legal_actions
            and _is_realtime(state)
            and _picked_from_legal_actions(exc)
        ):
            raise DecisionError(_cant_play_realtime_message(context)) from exc
        raise DecisionError(
            f"choose_action raised {exc!r} for session {context.session_id!r}."
        ) from exc


def _invoke_message_decision(
    decision_fn: MessageDecisionFn, state: GameState, context: DecisionContext
) -> Any:
    try:
        return decision_fn(state, context)
    except Exception as exc:
        raise DecisionError(
            f"choose_message raised {exc!r} for session {context.session_id!r}."
        ) from exc


def _is_realtime(state: GameState) -> bool:
    """Whether the match runs in real time: the state's ``pacing.mode``."""
    pacing = state.raw.get("pacing")
    return isinstance(pacing, dict) and pacing.get("mode") == _REALTIME_PACING


def _resolve_result_hook(choose_action: Any) -> Callable[[dict, DecisionContext], Any] | None:
    """An optional ``on_action_result(result, context)`` on the agent object
    (resolved from the original value, like ``choose_message``)."""
    method = getattr(choose_action, "on_action_result", None)
    return method if callable(method) else None


def _report_result(hook, result: dict, context: DecisionContext) -> None:
    if hook is None:
        return
    try:
        hook(result, context)
    except Exception as exc:
        raise DecisionError(
            f"on_action_result raised {exc!r} for session {context.session_id!r}."
        ) from exc


def _with_game_config(game: Any, context: DecisionContext) -> DecisionContext:
    """``context`` with ``game_config`` from ``get_game_config``, fetched once
    and best-effort: a server or session without it leaves ``None``."""
    getter = getattr(game, "get_game_config", None)
    if context.game_config is not None or not context.game_type or not callable(getter):
        return context
    try:
        config = getter(context.game_type)
    except AltruAgentError as exc:
        # Best-effort: also a temporary failure of the seat's re-grant, which
        # arrives as a control-plane error rather than an MCPToolError.
        if isinstance(exc, MCPToolError) or _is_transient(exc):
            return context
        raise
    return replace(context, game_config=config) if isinstance(config, dict) else context


def _validate_decision(decision: Any, legal_actions: list[LegalAction]) -> Union[_Resign, _Wait, _PlayAction]:
    if isinstance(decision, WithReasoning):
        inner = decision.action
        if inner is RESIGN or inner is WAIT or isinstance(inner, WithReasoning):
            raise DecisionError(
                "WithReasoning(...) must wrap a move (a LegalAction, action_id, "
                f"int, or structured dict), not {inner!r}."
            )
        return _validate_decision(inner, legal_actions)._replace(
            reasoning_summary=decision.reasoning_summary
        )

    if decision is RESIGN:
        return RESIGN

    if decision is WAIT:
        return WAIT

    # bool is a subclass of int in Python (isinstance(True, int) is True) —
    # reject it explicitly before the int branch below would otherwise treat
    # it as a legal action lookup.
    if isinstance(decision, bool):
        raise DecisionError(
            "choose_action must return a LegalAction, its action_id (str), a "
            "matching int, a structured dict action, or altruagent.RESIGN — "
            f"got {decision!r} (bool)."
        )

    if isinstance(decision, LegalAction):
        candidate = decision.action_id
    elif isinstance(decision, str):
        candidate = decision
    elif isinstance(decision, int):
        candidate = str(decision)
    elif isinstance(decision, dict):
        if not decision:
            raise DecisionError(
                "choose_action returned an empty dict — provide the structured "
                "action payload the server expects (e.g. "
                "{'type': 'submit_team', 'team': [...]})."
            )
        return _PlayAction(action_id=None, action=decision)
    else:
        raise DecisionError(
            "choose_action must return a LegalAction from state.legal_actions, "
            "its action_id (str), a matching int, a structured dict action, or "
            f"altruagent.RESIGN — got {decision!r} ({type(decision).__name__})."
        )

    matched = next((a for a in legal_actions if a.action_id == candidate), None)
    if matched is None:
        raise DecisionError(
            f"choose_action returned {decision!r}, which does not match any "
            f"current legal action's action_id (tried {candidate!r} against "
            f"{[a.action_id for a in legal_actions]!r}). An int is only valid "
            "for OpenSpiel-family games whose action_ids are stringified "
            "integers — never guessed/coerced for other games."
        )
    return _PlayAction(action_id=matched.action_id, action=None)


def _validate_message_decision(decision: Any) -> MessageDecision:
    if decision is TERMINATE_MESSAGING:
        return decision
    if not isinstance(decision, SendMessage):
        raise DecisionError(
            "choose_message must return an altruagent.SendMessage(...) or "
            f"altruagent.TERMINATE_MESSAGING — got {decision!r} "
            f"({type(decision).__name__})."
        )
    return decision


def _last_message_seq(state: GameState, extra: dict | None = None) -> int | None:
    """Highest message ``seq`` this agent has seen: the state's
    ``new_messages``, plus (after a ``send_message``) that call's own
    ``message``/``new_messages``. Passed to ``wait_for_update`` so a message
    that arrived before the wait started still wakes it.
    """
    seqs = [m.index for m in state.new_messages]
    if extra:
        messages = list(extra.get("new_messages") or [])
        if isinstance(extra.get("message"), dict):
            messages.append(extra["message"])
        seqs += [m["seq"] for m in messages if isinstance(m, dict) and isinstance(m.get("seq"), int)]
    return max(seqs) if seqs else None


def _is_unknown_tool_error(exc: MCPToolError) -> bool:
    """A server predating ``wait_for_update`` rejects it at the protocol
    level (FastMCP's ``Unknown tool: ...``), with no error_code.
    """
    return exc.error_code is None and "unknown tool" in str(exc).lower()


def _is_transient(exc: BaseException) -> bool:
    """A temporary failure worth retrying in place (see
    ``TRANSIENT_GIVE_UP_SECONDS``). A server without ``wait_for_update``
    also answers with no error code, but that is permanent: never retried.
    """
    if isinstance(exc, MCPToolError):
        if _is_unknown_tool_error(exc):
            return False
        if exc.error_code == _WAIT_BUSY_ERROR_CODE:
            return True
    return is_transient_error(exc)


def _is_tool_failure(exc: BaseException) -> bool:
    """The game server answered the call with an MCP tool error
    (``MCPToolError.protocol_error``: arguments it couldn't accept, or the
    tool crashed). Not a connection problem. After a move or message (whose
    content comes from the agent) it is most likely to fail the same way
    again, so it is tried once more only, in case it was a one-off (the move
    decided again from a fresh read). After a read, whose arguments the runner
    builds itself, it is the server's own trouble (GameAPI answers every game
    error as a normal result), so it is retried like a connection problem. A
    missing ``wait_for_update`` is handled on its own."""
    return (
        isinstance(exc, MCPToolError)
        and getattr(exc, "protocol_error", False)
        and not _is_unknown_tool_error(exc)
    )


# A move or message whose content can't be sent as JSON (``MCPToolError.
# local_error``) is a bug in the agent's own code: it fails at once, like an
# invalid move, never as a connection problem.
_UNSENDABLE_HINT = (
    "Use plain Python values only: str, int, float, bool, None, lists and dicts "
    "(for example int(x) for a numpy number)."
)


def _is_unsendable(exc: BaseException) -> bool:
    return isinstance(exc, MCPToolError) and getattr(exc, "local_error", False)


def _is_retried(exc: BaseException) -> bool:
    """Retried in place: a temporary failure, or a first tool failure."""
    return _is_transient(exc) or _is_tool_failure(exc)


class _TransientRetry:
    """Paces the retries after temporary failures in one game, and gives up
    (re-raising the last error) once failures have lasted
    ``TRANSIENT_GIVE_UP_SECONDS`` without a single successful call. A tool
    failure (``_is_tool_failure``) of a move or message is retried once, and
    re-raised if it happens again before a call succeeds; one of a read is
    retried like a temporary failure.
    """

    def __init__(self, *, sleep: Callable[[float], None], now: Callable[[], float],
                 log: Callable[[str], None]) -> None:
        self._sleep = sleep
        self._now = now
        self._log = log
        self._since: float | None = None
        self._back = ""
        self._streak = 0
        self._tool_failure_retried = False

    def failed(self, exc: BaseException, *, read: bool = False) -> None:
        """Pause before the next try, or re-raise ``exc`` if it's time to give
        up. ``read``: the call was a read the runner built itself."""
        current = self._now()
        if _is_tool_failure(exc) and not read:
            if self._tool_failure_retried:
                raise exc
            self._tool_failure_retried = True
            self._log(f"The game server couldn't handle that call ({exc}); trying once more.")
        elif self._since is None:
            self._since = current
            if _is_tool_failure(exc):
                self._back = "The game server answers again; the game goes on."
                self._log(f"The game server couldn't answer ({exc}); retrying for up to "
                          f"{TRANSIENT_GIVE_UP_SECONDS:.0f}s.")
            else:
                self._back = "Connection back; the game goes on."
                self._log(f"Connection problem ({exc}); retrying for up to {TRANSIENT_GIVE_UP_SECONDS:.0f}s.")
        elif current - self._since >= TRANSIENT_GIVE_UP_SECONDS:
            raise exc
        delay = min(TRANSIENT_RETRY_MAX_SECONDS, TRANSIENT_RETRY_FIRST_SECONDS * 2 ** self._streak)
        self._streak += 1
        self._sleep(delay * random.uniform(0.5, 1.0))

    def succeeded(self) -> None:
        if self._since is not None:
            self._log(self._back)
        self._since = None
        self._streak = 0
        self._tool_failure_retried = False


def _resign_left_game_running(result: Any) -> bool:
    """Whether a ``resign`` answer says the game goes on without this agent.

    Werewolf (since 2026-10-07): a resign takes one player out and the other
    six play on, so the answer has ``is_terminal: false`` (and ``eliminated:
    true``). Every other game ends on a resign, and so does an answer without
    ``is_terminal``.
    """
    return isinstance(result, dict) and result.get("is_terminal") is False


def _catch_up_messages(game: Any, state: GameState, seen_seq: int | None) -> None:
    """Add to ``state.new_messages`` what was said after this agent's last read but before the
    phase flipped: ``new_messages`` restarts at each flip, so in Werewolf the chat that ends a
    discussion never reached the vote (or, for the last day, the end of the game). Fetched with
    ``get_messages(since=...)``; only messages this seat may see are added. Best effort: any
    failure leaves the state as it was."""
    if seen_seq is None:
        return
    try:
        answer = game.get_messages(since=seen_seq)
        items = answer if isinstance(answer, list) else next(
            (answer[k] for k in ("messages", "new_messages", "items") if isinstance(answer.get(k), list)), [])
        me = state.raw.get("your_position")
        have = {m.index for m in state.new_messages}
        extra = []
        for item in items:
            if not isinstance(item, dict):
                continue
            message = Message.from_dict(item)
            visible = not message.recipients or me is None or me in message.recipients or message.sender == me
            if message.index > seen_seq and message.index not in have and visible:
                extra.append(message)
                have.add(message.index)
        if extra:
            state.new_messages = sorted(list(state.new_messages) + extra, key=lambda m: m.index)
    except Exception:  # noqa: BLE001 - a nicety; never cost the game
        return


def _max_seq(state: GameState, seen_seq: int | None) -> int | None:
    seqs = [m.index for m in state.new_messages] + ([seen_seq] if seen_seq is not None else [])
    return max(seqs) if seqs else None


def _terminal_game_state(last_state: GameState, result: dict) -> GameState:
    """Merge a ``get_result()``/``resign()`` result (authoritative for
    ``returns``/``termination_reason``, confirmed absent from ``get_state()``/
    ``play_action()``) onto the last known state's raw dict (for ``phase``/
    ``observation`` context, still valid once terminal).
    """
    return GameState.from_mcp_state(last_state.raw, result=result)


def run_game(
    game: MCPGameSession,
    context: DecisionContext,
    choose_action: DecisionFn,
    *,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> GameState:
    """Play one already-open ``MCPGameSession`` to completion.

    Loop, once per iteration:

    1. Read state (``get_game_state``; afterwards each step below hands back
       the next state itself).
    2. If terminal, fetch the result (``get_result``) and return the merged
       final state.
    3. If this agent was eliminated (Werewolf), it can no longer act or
       chat: wait (step 6) until the game ends.
    4. If ``phase == "messaging"``: resolve an optional ``choose_message``,
       validate the result, and submit it (``send_message`` with
       ``message_type="chat"`` or ``"terminate"``). If the round hasn't
       actually advanced yet (``send_message``'s own result still reports
       ``phase != "moving"`` — e.g. this agent terminated but others
       haven't), wait (step 6) for the next message or the phase change.
    5. Else if ``is_current_actor``: take ``legal_actions`` from the state
       (the server embeds them when this agent can act; ``get_legal_actions``
       is only a fallback when it didn't), invoke ``choose_action``, validate
       the result, and submit it (``play_action`` for a matched/structured
       action, ``resign`` for ``RESIGN``). ``play_action``'s result carries
       the post-move state, which becomes the next state directly. A resign
       that ends the game returns at once; in Werewolf, where it takes only
       this agent out and the game goes on, the runner re-reads the state
       (now ``eliminated``) and waits for the end as in step 3.
    6. Else (not this agent's turn): ``wait_for_update`` — returns as soon as
       anything changes, or after ``WAIT_FOR_UPDATE_TIMEOUT_SECONDS`` with
       the unchanged state (the loop just waits again). Against a server without that tool,
       sleeps ``wait_seconds`` and re-reads instead.

    In a real-time game (``pacing.mode == "realtime"``), ``WAIT`` waits as
    in step 6, ``INVALID_ACTION`` re-reads the state like a race, and
    ``RUNTIME_TEMPORARILY_UNAVAILABLE`` re-reads it after
    ``REALTIME_RETRY_SECONDS``; ``context.game_config`` is fetched once
    before the first decision. ``on_action_result`` (optional, on the agent
    object) gets every ``play_action`` answer and every recovered refusal.

    A race error (``STALE_STATE``/``NOT_YOUR_TURN``/``WRONG_PHASE``/
    ``GAME_ALREADY_COMPLETE``/``PLAYER_ELIMINATED`` — the state changed
    between our last read and this submit, not a contestant bug) refetches
    state and continues. ``RUNTIME_UNAVAILABLE`` (the adapter doesn't
    actually support a capability the runner expected) raises
    ``UnsupportedGameFlowError``. A contestant-caused messaging/action error
    raises ``DecisionError``.

    A temporary failure of any call (``errors.is_transient_error``: a
    dropped connection, a gateway or server error, a busy game engine; or
    ``TOO_MANY_WAITS`` while an abandoned wait still runs on the server) is
    retried in place: a read is simply tried again after a short pause; a
    move or message is never resent — the runner pauses, re-reads the state
    and decides again. ``log`` gets one line when the trouble starts and one
    when it ends. Only after ``TRANSIENT_GIVE_UP_SECONDS`` without a single
    successful call does the error propagate. A read the server answers with
    an MCP tool error (``MCPToolError.protocol_error``: its own trouble) is
    retried the same way. A move or message answered that way is retried once
    (decided again from a fresh read); a second one before any call succeeds
    propagates. A move or message that can't be sent as JSON
    (``MCPToolError.local_error``) raises ``DecisionError`` at once. Any other
    error propagates immediately.

    Returns the final ``GameState`` once the match is terminal.
    """
    decision_fn = _resolve_decision_fn(choose_action)
    message_decision_fn = _resolve_message_decision_fn(choose_action)
    result_hook = _resolve_result_hook(choose_action)
    long_poll_supported = True
    config_fetched = False
    retry = _TransientRetry(sleep=sleep, now=now, log=log)

    def read(call: Callable[[], Any], *, settles: bool = True) -> Any:
        """One read of the game (safe to repeat), retried in place after a
        temporary failure. ``settles=False``: the read alone doesn't end a
        spell of trouble (see ``recover``)."""
        while True:
            try:
                answer = call()
            except AltruAgentError as exc:
                if not _is_retried(exc):
                    raise
                retry.failed(exc, read=True)
                continue
            if settles:
                retry.succeeded()
            return answer

    def recover(exc: AltruAgentError) -> GameState:
        """After a temporary failure of a move or message: pause, then read
        the state again. Never resend: it may have landed. The re-read doesn't
        end the trouble; a move or message that goes through, or a wait (the
        game moved on), does. So a move that keeps failing while reads work is
        retried at a slowing pace and reaches the worker after
        ``TRANSIENT_GIVE_UP_SECONDS``, instead of being retried every second
        for the rest of the game."""
        retry.failed(exc)
        return read(game.get_state, settles=False)

    def wait(current: GameState, message_seq: int | None) -> GameState:
        nonlocal long_poll_supported
        if long_poll_supported:
            try:
                return read(
                    lambda: game.wait_for_update(
                        since_version=current.state_version,
                        since_message_seq=message_seq,
                        since_is_current_actor=bool(current.is_current_actor),
                        since_phase=current.phase,
                        timeout_seconds=WAIT_FOR_UPDATE_TIMEOUT_SECONDS,
                    )
                )
            except MCPToolError as exc:
                if not _is_unknown_tool_error(exc):
                    raise
                long_poll_supported = False
        sleep(wait_seconds)
        return read(game.get_state)

    state = read(game.get_state)
    seen_seq: int | None = None  # highest message seq shown to the agent so far

    while True:
        if state.is_terminal:
            result = read(game.get_result)
            final = _terminal_game_state(state, result)
            final.new_messages = []
            _catch_up_messages(game, final, seen_seq)  # the last day's closing chat, for on_game_end
            return final

        if state.raw.get("eliminated"):
            state = wait(state, _last_message_seq(state))
            continue

        if state.phase == _MESSAGING_PHASE:
            seen_seq = _max_seq(state, seen_seq)
            decision = _validate_message_decision(
                _invoke_message_decision(message_decision_fn, state, context)
            )
            try:
                if decision is TERMINATE_MESSAGING:
                    result = game.send_message(message_type="terminate")
                else:
                    result = game.send_message(
                        message_type="chat",
                        content=decision.content,
                        recipients=decision.recipients,
                    )
            except AltruAgentError as exc:
                code = exc.error_code if isinstance(exc, MCPToolError) else None
                if code in _RACE_ERROR_CODES:
                    state = read(game.get_state)
                    continue
                if code == _CAPABILITY_ERROR_CODE:
                    raise UnsupportedGameFlowError(
                        f"Match {context.session_id!r} reported phase="
                        f"{_MESSAGING_PHASE!r}, but its adapter does not support "
                        "messaging tools — a runner/adapter mismatch, not a "
                        "contestant bug."
                    ) from exc
                if code in _MESSAGE_SCOPED_ERROR_CODES:
                    raise DecisionError(
                        f"choose_message produced an invalid messaging action "
                        f"for session {context.session_id!r}: {exc}"
                    ) from exc
                if _is_unsendable(exc):
                    raise DecisionError(
                        f"choose_message produced a message that can't be sent "
                        f"for session {context.session_id!r} ({exc}). {_UNSENDABLE_HINT}"
                    ) from exc
                if _is_retried(exc):
                    state = recover(exc)
                    continue
                raise
            retry.succeeded()
            # send_message's own result already carries the post-call phase.
            # Terminating is idempotent server-side, so a contestant that
            # already terminated this round (or the default auto-terminate)
            # would just get a no-op success back while the others are still
            # talking — wait for the next message or the phase flip instead
            # of re-sending.
            if result.get("phase") == "moving":
                state = read(game.get_state)
            else:
                state = wait(state, _last_message_seq(state, result))
            continue

        if state.is_current_actor:
            realtime = _is_realtime(state)
            if realtime and not config_fetched:
                config_fetched = True
                context = _with_game_config(game, context)
            if state.raw.get("legal_actions") is None:
                # The server omits legal_actions when a move landed between
                # its state and legal-actions reads; fetch them directly.
                legal = read(game.get_legal_actions, settles=False)  # the move that follows settles
                state.legal_actions = [
                    LegalAction.from_dict(a) for a in legal.get("actions") or []
                ]
                state.state_version = int(legal.get("state_version", state.state_version))

            _catch_up_messages(game, state, seen_seq)
            seen_seq = _max_seq(state, seen_seq)
            decision = _validate_decision(
                _invoke_decision(decision_fn, state, context), state.legal_actions
            )
            if decision is WAIT:
                if not realtime:
                    raise DecisionError(
                        "choose_action returned altruagent.WAIT in a turn-based game "
                        f"(session {context.session_id!r}); WAIT is only for real-time "
                        "games, where nothing needs to be sent right now."
                    )
                state = wait(state, _last_message_seq(state))
                continue
            left_the_game = False
            try:
                if decision is RESIGN:
                    result = game.resign()
                    if not _resign_left_game_running(result):
                        return _terminal_game_state(state, result)
                    left_the_game = True
                else:
                    # Only passed when the contestant supplied one, so agents
                    # that never use WithReasoning send exactly what they did before.
                    reasoning = (
                        {"reasoning_summary": decision.reasoning_summary}
                        if decision.reasoning_summary
                        else {}
                    )
                    result = game.play_action(
                        action_id=decision.action_id,
                        action=decision.action,
                        state_version=state.state_version,
                        **reasoning,
                    )
            except AltruAgentError as exc:
                code = exc.error_code if isinstance(exc, MCPToolError) else None
                recovered = code in _RACE_ERROR_CODES or (
                    realtime
                    and code in _ACTION_SCOPED_ERROR_CODES | _REALTIME_TRANSIENT_ERROR_CODES
                )
                if recovered:
                    _report_result(result_hook, {"error": code, "detail": str(exc)}, context)
                    if code in _REALTIME_TRANSIENT_ERROR_CODES:
                        sleep(REALTIME_RETRY_SECONDS)
                    state = read(game.get_state)
                    continue
                if code == _CAPABILITY_ERROR_CODE:
                    raise UnsupportedGameFlowError(
                        f"Match {context.session_id!r} reported it was this "
                        "agent's turn, but its adapter rejected the move "
                        "capability — a runner/adapter mismatch, not a "
                        "contestant bug."
                    ) from exc
                if code in _ACTION_SCOPED_ERROR_CODES:
                    raise DecisionError(
                        f"choose_action produced an invalid action for session "
                        f"{context.session_id!r}: {exc}"
                    ) from exc
                if _is_unsendable(exc):
                    raise DecisionError(
                        f"choose_action returned a move that can't be sent "
                        f"for session {context.session_id!r} ({exc}). {_UNSENDABLE_HINT}"
                    ) from exc
                if _is_retried(exc):
                    # Not resent: it may have landed. The fresh state says
                    # whether a decision is still due.
                    if code is not None:
                        _report_result(result_hook, {"error": code, "detail": str(exc)}, context)
                    state = recover(exc)
                    continue
                raise
            retry.succeeded()
            if left_the_game:
                # Werewolf: the resign took only this agent out and the game
                # goes on. The fresh state says `eliminated`, so the loop just
                # waits for the end (step 3) and returns the final result.
                state = read(game.get_state)
                continue
            if isinstance(result, dict):
                _report_result(result_hook, result, context)
            # While the game continues, play_action returns the post-move
            # state; once it's over (or on an older server or in a real-time
            # game, whose answer carries verdicts only) it doesn't.
            post_move = result.get("state") if isinstance(result, dict) else None
            state = (
                GameState.from_mcp_state(post_move)
                if isinstance(post_move, dict)
                else read(game.get_state)
            )
            continue

        state = wait(state, _last_message_seq(state))


def run_match(
    match: Match,
    agent_id: str,
    choose_action: DecisionFn,
    *,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> GameState:
    """Play one ``Match`` (as returned by ``AltruAgentClient.sessions()``)
    to completion, using ``choose_action`` for every decision.

    A thin convenience wrapper around ``run_game``: opens the match via
    ``match.game()`` — the production ``MCPGameSession`` path (reusing its
    existing lazy ``game_server_url`` resolution and caching rather than
    duplicating that logic here; never ``match.rest_game()``, the debug-only
    REST path) — and builds the ``DecisionContext`` from the ``Match``'s own
    fields plus the caller-supplied ``agent_id`` (a ``Match`` has no notion
    of "which agent is playing it").
    """
    game = match.game()
    context = DecisionContext(
        session_id=match.session_id,
        tournament_id=match.tournament_id,
        game_type=match.game_type,
        agent_id=agent_id,
    )
    return run_game(game, context, choose_action, wait_seconds=wait_seconds, sleep=sleep)
