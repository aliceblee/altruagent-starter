"""Per-match worker process — the child side of Milestone 4C's
process-per-match concurrency model.

``run_worker`` and ``_process_entry`` must stay top-level, importable
functions (never a closure/lambda/bound method) — Windows has no ``fork``;
``multiprocessing``'s ``spawn`` start method launches a genuinely fresh
Python interpreter for every worker and re-imports this module by name to
find its target, rather than copying the parent's memory. That's also
exactly what gives each worker its own isolated module-level state for
free: a contestant's ``agent.agent`` module is reimported from scratch in
every worker process, so even careless module-level globals (not just a
``create_agent()``-returned instance) never leak between matches.

Only primitive, picklable data crosses the process boundary (``WorkerInput``
below) — confirmed empirically during the 4C design audit that
``AltruAgentClient``/``GameSession``/a client-attached ``Match`` are not
picklable (``httpx.Client`` holds a real ``_thread.RLock``), so nothing here
ever tries to pass one. Each worker builds its own ``AltruAgentClient()``
from its own inherited environment/`.env` (the same config-loading path
every other entry point already uses) rather than receiving the API key
through ``WorkerInput`` at all.
"""

from __future__ import annotations

import os
import random
import time
from functools import partial
from typing import TYPE_CHECKING, Callable, NamedTuple

from .client import AltruAgentClient
from .console import print_line
from .errors import AuthenticationError, PlatformError, is_transient_error
from .mcp_game import MCPGameSession
from .models import Match
from .runner import DecisionError, UnsupportedGameFlowError, _is_transient
from .runner import run_match as _default_run_match

if TYPE_CHECKING:
    from .models import GameState
    from .official import OfficialAgentClient

# The entire parent<->child result-signaling mechanism (see
# supervisor.py) — no Queue/Pipe is used; a worker always returns one of
# these via its process exit code. EXIT_MATCH_FAILURE and EXIT_UNEXPECTED
# both lead to the same cooldown treatment in the supervisor; the
# distinction exists purely so logs can say which kind of failure occurred.
EXIT_SUCCESS = 0
EXIT_MATCH_FAILURE = 1
EXIT_UNEXPECTED = 2
# Tournament workers only: another runtime holds this seat's execution lease.
# The supervisor retries the seat once a lease could have lapsed.
EXIT_SEAT_BUSY = 3
# Tournament workers only: a temporary problem (the platform busy, a dropped
# connection) kept this worker from ever reaching its game. Nothing was
# played, so the supervisor tries again within seconds while the game's
# connect window is open: a short hiccup must never become a no-show.
EXIT_NOT_CONNECTED = 4
# Tournament workers only: the platform says the event registration isn't
# complete (for example the Official Rules were updated and must be accepted
# again). The worker has printed what to do; the supervisor keeps trying.
EXIT_REGISTRATION_INCOMPLETE = 5
# Tournament workers only: the worker reached its game, then every call failed
# for a temporary reason for runner.TRANSIENT_GIVE_UP_SECONDS (a lasting
# outage or a dropped connection), so the runner gave up. Not the agent's
# fault: the supervisor picks the game up again soon if it had been playing.
EXIT_CONNECTION_LOST = 6
# Tournament workers only: the platform refused the Official Agent Key itself
# (``official.is_fatal_auth_error``: it was rotated or revoked, or the agent
# is no longer Self-hosted). No retry can help until the contestant puts the
# new key in .env and restarts, so the supervisor starts no new workers.
EXIT_KEY_REFUSED = 7

# Before the game is reached, a temporary failure to get the seat's grant is
# retried inside the worker (no new process, no new agent): after about 3 s,
# then 6 s, 12 s, and every 15 s, with jitter, for up to this long. Then the
# worker exits with EXIT_NOT_CONNECTED and the supervisor starts a new one.
FIRST_GRANT_RETRY_SECONDS = 60.0
FIRST_GRANT_RETRY_FIRST_SECONDS = 3.0
FIRST_GRANT_RETRY_MAX_SECONDS = 15.0

_MATCH_SCOPED_ERRORS = (DecisionError, UnsupportedGameFlowError, PlatformError)


class WorkerInput(NamedTuple):
    """Primitive, picklable description of one match for a worker process to
    reconstruct and play — exactly the fields ``DecisionContext`` needs.
    Deliberately does not include ``control_url``/``api_key``: the worker
    builds its own client from its own inherited environment instead (see
    module docstring).
    """

    session_id: str
    tournament_id: str | None
    game_type: str | None
    agent_id: str


def run_worker(
    worker_input: WorkerInput,
    *,
    client_factory: Callable[[], AltruAgentClient] = AltruAgentClient,
    match_factory: Callable[..., Match] = Match.from_dict,
    agent_module: object | None = None,
    run_match_fn: Callable[..., "GameState"] = _default_run_match,
) -> int:
    """Play exactly one match: build a client, reconstruct its ``Match``,
    call ``agent.agent.create_agent()`` exactly once, and hand the result to
    the existing, unmodified ``run_match``.

    Returns an exit code (``EXIT_*`` above) rather than raising — this is
    what ``_process_entry`` turns into a real process exit code, and it's
    also why this function is safe and useful to call directly (no real
    process involved) from tests: every failure path is captured as a
    return value, never an uncaught exception escaping to the caller.

    ``client_factory``/``match_factory``/``agent_module``/``run_match_fn``
    all default to the real production pieces; tests override them to avoid
    any real network call or dependency on a real ``agent/agent.py``.
    """
    pid = os.getpid()
    print_line(f"[worker pid={pid}] starting session_id={worker_input.session_id}")

    client: AltruAgentClient | None = None
    try:
        if agent_module is None:
            import agent.agent as agent_module  # the contestant's own code

        create_agent = getattr(agent_module, "create_agent", None)
        if not callable(create_agent):
            print_line(
                f"[worker pid={pid}] agent.agent.create_agent is missing or "
                "not callable — nothing to play this match with."
            )
            return EXIT_UNEXPECTED

        client = client_factory()
        match = match_factory(
            {
                "session_id": worker_input.session_id,
                "tournament_id": worker_input.tournament_id,
                "game_type": worker_input.game_type,
                "status": "in_progress",
            },
            client=client,
        )

        contestant = create_agent()
        final_state = run_match_fn(match, worker_input.agent_id, contestant)
        print_line(
            f"[worker pid={pid}] match finished session_id={worker_input.session_id} "
            f"termination_reason={final_state.termination_reason}"
        )
        return EXIT_SUCCESS
    except KeyboardInterrupt:
        # Windows delivers Ctrl+C to the whole console process group, so
        # this worker sees it directly too, independent of whether the
        # parent's own cleanup reaches it in time. Exit quietly — no
        # traceback, no resign, no further API calls.
        print_line(f"[worker pid={pid}] interrupted, stopping")
        return EXIT_SUCCESS
    except _MATCH_SCOPED_ERRORS as exc:
        print_line(
            f"[worker pid={pid}] match failed session_id={worker_input.session_id}: {exc}"
        )
        return EXIT_MATCH_FAILURE
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see EXIT_UNEXPECTED
        print_line(f"[worker pid={pid}] unexpected error: {exc!r}")
        return EXIT_UNEXPECTED
    finally:
        if client is not None:
            client.close()


# How a game's last line names this agent's result. The game's own result for
# this player comes first (Werewolf), then the winner (Pokémon, Red Alert),
# then the score, read the way the platform reads it: above 0.5 a win, below
# a loss, exactly 0.5 a draw.
_OUTCOME_WORDS = {"win": "won", "loss": "lost", "draw": "drew", "resigned": "resigned", "timed_out": "timed out"}


def _number(value) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _score(final_state, agent_id: str) -> float | None:
    result = getattr(final_state, "final_result", None) or {}
    score = _number(result.get("your_return"))
    if score is None:
        score = _number((getattr(final_state, "returns", None) or {}).get(agent_id))
    return score


def game_outcome(final_state, agent_id: str) -> str | None:
    """This agent's result in a finished game: ``"won"``, ``"lost"``,
    ``"drew"``, ``"resigned"``, ``"timed out"``, ``"no result"`` (the server
    couldn't finish the game), or ``None`` when the server didn't say."""
    result = getattr(final_state, "final_result", None) or {}
    structured = result.get("result")
    players = structured.get("seats") if isinstance(structured, dict) else None
    mine = players.get(agent_id) if isinstance(players, dict) else None
    if isinstance(mine, dict) and mine.get("outcome") in _OUTCOME_WORDS:
        return _OUTCOME_WORDS[mine["outcome"]]
    if result.get("status") == "failed" or result.get("failure_reason"):
        return "no result"
    winner = result.get("winner_agent_id")
    if winner:
        return "won" if winner == agent_id else "lost"
    score = _score(final_state, agent_id)
    if score is None:
        return None
    return "won" if score > 0.5 else "lost" if score < 0.5 else "drew"


def _report_game_end(contestant, final_state, context, outcome, print_line, label: str) -> None:
    """An optional ``on_game_end(final_state, context, outcome)`` on the agent object, called once
    after the game ends (``outcome`` as in ``game_outcome``). It can't affect the game, so a hook
    that raises is reported and otherwise ignored."""
    method = getattr(contestant, "on_game_end", None)
    if not callable(method):
        return
    try:
        method(final_state, context, outcome)
    except Exception as exc:  # noqa: BLE001 - the game is over; never fail the worker for this
        print_line(f"{label} on_game_end raised {exc!r}")


def finished_line(final_state, agent_id: str, seat_position: int | None = None) -> str:
    """The worker's last line for a game that ended: how this agent did, when
    the server said, then the technical detail. For example
    ``finished: your agent (player 2) won (termination_reason=completed, score=1.0)``."""
    score = _score(final_state, agent_id)
    detail = f"termination_reason={getattr(final_state, 'termination_reason', None)}" + (
        f", score={score}" if score is not None else "")
    outcome = game_outcome(final_state, agent_id)
    if outcome is None:
        return f"finished ({detail})"
    if outcome == "no result":
        return f"finished with no result ({detail})"
    player = f" (player {seat_position + 1})" if isinstance(seat_position, int) else ""
    return f"finished: your agent{player} {outcome} ({detail})"


class TournamentWorkerInput(NamedTuple):
    """Primitive, picklable description of one assigned seat (a Testing or
    tournament game). Carries no credential: the worker reads the Official
    Agent Key from its own inherited environment/.env, exactly like
    ``WorkerInput`` above.
    """

    seat_id: str
    match_id: str
    game_type: str | None
    agent_spec: str
    # The supervisor's one execution id, shared by all of its workers (the
    # seat lease owner). Kept out of repr so it never lands in a log.
    execution_id: str
    # The assignment's tournament, if the server named one (None for Testing);
    # handed to the contestant as DecisionContext.tournament_id.
    tournament_id: str | None = None
    # The supervisor's current agent session token, so the worker needn't
    # sign in again (each sign-in costs the platform one anonymous sign-in).
    # Handed over the process pipe only; kept out of repr like execution_id.
    # None (or an expired one) just means the worker signs in itself.
    agent_session: str | None = None

    def __repr__(self) -> str:
        return (
            f"TournamentWorkerInput(seat_id={self.seat_id!r}, match_id={self.match_id!r}, "
            f"game_type={self.game_type!r}, agent_spec={self.agent_spec!r}, "
            f"tournament_id={self.tournament_id!r})"
        )


def run_tournament_worker(
    worker_input: TournamentWorkerInput,
    *,
    official_factory: Callable[[], "OfficialAgentClient"] | None = None,
    run_game_fn: Callable[..., "GameState"] | None = None,
    lease_renew_seconds: float | None = None,
    lease_retry_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> int:
    """Play one assigned seat: build the contestant from ``agent_spec``
    (``create_agent()`` exactly once, in this process), request the seat's
    SeatGrant, and hand the seat to the existing ``run_game`` loop. GameAPI
    only ever sees the SeatGrant token; renewing it after a 401 re-requests
    the grant (``OfficialSeatAuth``). While the match runs, a
    ``SeatLeaseKeeper`` keeps this runtime's execution lease on the seat; if
    another runtime takes it (``seat_busy``), every further gameplay call is
    refused and the worker exits with ``EXIT_SEAT_BUSY``. Returns an
    ``EXIT_*`` code.

    A temporary failure to get the first grant is retried here for up to
    ``FIRST_GRANT_RETRY_SECONDS`` (``run_game`` retries temporary failures
    during play itself). A worker that never reached its game because of a
    temporary problem exits with ``EXIT_NOT_CONNECTED``; one that reached it
    and then lost the connection for good (``run_game`` gave up on temporary
    failures) exits with ``EXIT_CONNECTION_LOST``; one refused because the
    event registration isn't complete exits with
    ``EXIT_REGISTRATION_INCOMPLETE`` after printing what to do; one whose
    Official Agent Key the platform refused exits with ``EXIT_KEY_REFUSED``.
    """
    from .agent_loader import load_agent_factory
    from .models import DecisionContext
    from .official import (
        LEASE_RENEW_SECONDS,
        LEASE_RETRY_SECONDS,
        OfficialAgentClient,
        OfficialAgentError,
        OfficialSeatAuth,
        SeatLeaseKeeper,
        SeatLeaseLost,
        is_fatal_auth_error,
        is_registration_incomplete,
        registration_wait_message,
    )
    from .runner import _resolve_decision_fn, run_game

    pid = os.getpid()
    label = f"[match {worker_input.game_type or 'unknown'} seat={worker_input.seat_id}]"
    official_factory = official_factory or OfficialAgentClient
    run_game_fn = run_game_fn or partial(run_game, log=lambda message: print_line(f"{label} {message}"))
    print_line(f"{label} starting (pid={pid})")

    official = None
    game_client = None
    keeper = None
    game = None
    try:
        contestant = load_agent_factory(worker_input.agent_spec)()
        _resolve_decision_fn(contestant)

        official = official_factory()
        adopt = getattr(official, "use_session_token", None)
        if worker_input.agent_session and callable(adopt):
            adopt(worker_input.agent_session)  # no second sign-in for this game
        seat_auth = OfficialSeatAuth(official, worker_input.seat_id, worker_input.execution_id)
        game_client = AltruAgentClient(control_url=official.control_url, auth=seat_auth, load_env_file=False)
        give_up_at = now() + FIRST_GRANT_RETRY_SECONDS
        attempt = 0
        while True:
            try:
                game_client.login()  # the first grant for this seat
                break
            except (PlatformError, AuthenticationError) as exc:
                code = exc.error_code if isinstance(exc, OfficialAgentError) else None
                if code == "assignment_not_grantable":
                    print_line(f"{label} assignment already ended; nothing to play")
                    return EXIT_SUCCESS
                if code == "seat_busy":
                    print_line(f"{label} seat is held by another runtime; not playing it")
                    return EXIT_SEAT_BUSY
                if not is_transient_error(exc) or now() >= give_up_at:
                    raise
                error = exc
            if attempt == 0:
                print_line(f"{label} couldn't get into the game yet ({error}); trying again for up to "
                      f"{FIRST_GRANT_RETRY_SECONDS:.0f}s")
            delay = min(FIRST_GRANT_RETRY_MAX_SECONDS, FIRST_GRANT_RETRY_FIRST_SECONDS * 2 ** attempt)
            attempt += 1
            sleep(delay * random.uniform(0.75, 1.25))
        grant = seat_auth.grant

        keeper = SeatLeaseKeeper(
            official,
            worker_input.seat_id,
            worker_input.execution_id,
            renew_seconds=lease_renew_seconds or LEASE_RENEW_SECONDS,
            retry_seconds=lease_retry_seconds or LEASE_RETRY_SECONDS,
            log=lambda message: print_line(f"{label} {message}"),
        )
        keeper.start()
        game = _LeaseGuardedGameSession(
            game_client, lost=keeper.lost, session_id=grant.game_session_id, game_server_url=grant.gameapi_server_url
        )
        context = DecisionContext(
            session_id=grant.game_session_id,
            tournament_id=worker_input.tournament_id,
            game_type=grant.game_type,
            agent_id=grant.agent_id,
            seat_position=grant.seat_position,
        )
        final_state = run_game_fn(game, context, contestant)
        _report_game_end(contestant, final_state, context, game_outcome(final_state, grant.agent_id), print_line, label)
        print_line(f"{label} {finished_line(final_state, grant.agent_id, grant.seat_position)}")
        return EXIT_SUCCESS
    except KeyboardInterrupt:
        print_line(f"{label} interrupted, stopping")
        return EXIT_SUCCESS
    except SeatLeaseLost:
        print_line(f"{label} stopped: another runtime now holds this seat")
        return EXIT_SEAT_BUSY
    except Exception as exc:  # noqa: BLE001 - includes contestant factory errors and auth failures
        code = exc.error_code if isinstance(exc, (PlatformError, AuthenticationError)) else None
        if (keeper is not None and keeper.lost.is_set()) or (isinstance(exc, OfficialAgentError) and code == "seat_busy"):
            # Also when a GameAPI re-grant during play answered seat_busy.
            print_line(f"{label} stopped: another runtime now holds this seat")
            return EXIT_SEAT_BUSY
        if isinstance(exc, OfficialAgentError) and code == "assignment_not_grantable":
            print_line(f"{label} the game has ended; nothing more to play")
            return EXIT_SUCCESS
        if is_registration_incomplete(exc):
            print_line(f"{label} {registration_wait_message(exc)}")
            return EXIT_REGISTRATION_INCOMPLETE
        if is_fatal_auth_error(exc):
            # The key itself was refused (e.g. rotated while running).
            print_line(f"{label} {exc}")
            return EXIT_KEY_REFUSED
        if _is_transient(exc):
            if not getattr(game, "contacted", False):
                print_line(f"{label} couldn't reach the game because of a temporary problem ({exc}); trying again shortly")
                return EXIT_NOT_CONNECTED
            print_line(f"{label} lost the connection to the game ({exc}); reconnecting shortly")
            return EXIT_CONNECTION_LOST
        if isinstance(exc, _MATCH_SCOPED_ERRORS):
            print_line(f"{label} match failed: {exc}")
            return EXIT_MATCH_FAILURE
        print_line(f"{label} unexpected error: {exc!r}")
        return EXIT_UNEXPECTED
    finally:
        if keeper is not None:
            keeper.stop()
        for client in (game_client, official):
            if client is not None:
                client.close()


class _LeaseGuardedGameSession(MCPGameSession):
    """The seat's MCP session, refusing every further GameAPI call once the
    execution lease is lost (all tools go through ``_call``), so a worker that
    lost its seat can't keep acting on it.
    """

    def __init__(self, client, *, lost, **kwargs) -> None:
        super().__init__(client, **kwargs)
        self._lost = lost
        # True after the first call GameAPI answered: from then on the platform
        # counts this agent as connected to the game.
        self.contacted = False

    def _call(self, tool: str, arguments: dict) -> dict:
        if self._lost.is_set():
            from .official import SeatLeaseLost

            raise SeatLeaseLost("Another runtime now holds this seat's execution lease.")
        answer = super()._call(tool, arguments)
        self.contacted = True
        return answer


def _tournament_process_entry(worker_input: TournamentWorkerInput) -> None:
    """``multiprocessing.Process`` target for a tournament seat (see
    ``_process_entry``)."""
    raise SystemExit(run_tournament_worker(worker_input))


def _process_entry(worker_input: WorkerInput) -> None:
    """The actual ``multiprocessing.Process`` target. Top-level and
    importable by name, as Windows ``spawn`` requires. Translates
    ``run_worker``'s return value into a real process exit code via
    ``SystemExit`` — ``multiprocessing.Process.exitcode`` surfaces exactly
    this value to the parent once the process has exited.
    """
    raise SystemExit(run_worker(worker_input))
