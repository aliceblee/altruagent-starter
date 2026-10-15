"""Tests for the official tournament runtime: the supervisor loop
(``altruagent.supervisor.run_tournament_once``/``run_tournament_forever``)
with fake processes, and the per-seat worker
(``altruagent.worker.run_tournament_worker``) with a fake official client and
a recording ``run_game``. No real processes, network, or GameAPI.
"""

from __future__ import annotations

import sys
import textwrap
import types
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from altruagent.errors import AuthenticationError, PlatformError
from altruagent.mcp_game import MCPGameSession
from altruagent.mcp_transport import MCPToolError
from altruagent.models import OfficialAssignment, SeatGrant
from altruagent.official import AUTHENTICATE_PATH, OfficialAgentClient, OfficialAgentError
from altruagent.supervisor import (
    ALL_KINDS,
    KEY_REFUSED_MESSAGE,
    MIN_CONNECT_WINDOW_SECONDS,
    MISSING_POLLS_BEFORE_STOP,
    NOT_CONNECTED_RETRY_SECONDS,
    REGISTRATION_REMINDER_SECONDS,
    REGISTRATION_RETRY_SECONDS,
    RESTART_AFTER_PLAY_SECONDS,
    SEAT_FAILURE_BACKOFF_MAX_SECONDS,
    WORKER_PLAYED_SECONDS,
    TESTING,
    TEST_MATCH_WAITING_NOTE,
    TOURNAMENT,
    WAITING_MESSAGE,
    TournamentState,
    assignment_kind,
    describe_assignment,
    run_tournament_forever,
    run_tournament_once,
    tournament_waiting_warning,
)
from altruagent.runner import TRANSIENT_GIVE_UP_SECONDS, DecisionError
from altruagent.worker import (
    EXIT_CONNECTION_LOST,
    EXIT_KEY_REFUSED,
    EXIT_MATCH_FAILURE,
    EXIT_NOT_CONNECTED,
    EXIT_REGISTRATION_INCOMPLETE,
    EXIT_SEAT_BUSY,
    EXIT_SUCCESS,
    EXIT_UNEXPECTED,
    FIRST_GRANT_RETRY_SECONDS,
    TournamentWorkerInput,
    _tournament_process_entry,
    run_tournament_worker,
)
from test_official import CONTROL, KEY
from test_official import Backend as OfficialBackend
from test_official import assignment as official_assignment
from test_supervisor import FakeProcessFactory

SPEC = "agent.agent:create_agent"
EXEC = "exec-" + "q" * 40
NOW = datetime(2026, 10, 16, 15, 0, 0, tzinfo=timezone.utc)


def a(seat_id, match_id=None, game_type="pokemon_vgc_doubles_draft", **extra):
    return OfficialAssignment(match_id=match_id or f"match-{seat_id}", seat_id=seat_id, game_type=game_type,
                              seat_position=0, seat_count=2, match_status="in_progress", seat_status="running",
                              **extra)


def tournament_game(seat_id="seat-1", **overrides):
    fields = dict(
        game_type="werewolf", context="tournament", tournament_id="t-1", tournament_name="Fall Cup",
        round_label="Swiss round 1 of 3", opponents=("Alpha", "Beta"),
        connect_deadline_at="2026-10-16T15:03:40.000Z",
    )
    fields.update(overrides)
    return a(seat_id, **fields)


class FakeOfficial:
    """Returns scripted assignment listings (a list, or an exception)."""

    def __init__(self, *listings):
        self.listings = list(listings)
        self.calls = 0

    def assignments(self):
        self.calls += 1
        listing = self.listings.pop(0) if len(self.listings) > 1 else self.listings[0]
        if isinstance(listing, Exception):
            raise listing
        return listing


class Harness:
    def __init__(self, *listings, cooldown=60.0, kinds=ALL_KINDS):
        self.official = FakeOfficial(*listings)
        self.state = TournamentState()
        self.factory = FakeProcessFactory()
        self.log: list[str] = []
        self.clock = {"t": 0.0}
        self.cooldown = cooldown
        self.kinds = kinds

    def tick(self, advance=1.0):
        self.clock["t"] += advance
        run_tournament_once(
            self.official, self.state, agent_spec=SPEC, kinds=self.kinds, now=lambda: self.clock["t"],
            cooldown_seconds=self.cooldown, process_factory=self.factory, log=self.log.append,
            wall_now=lambda: NOW,
        )

    def started(self):
        return [p.args[0].seat_id for p in self.factory.processes]


# -- supervisor --------------------------------------------------------------------------


def test_no_assignments_starts_nothing_and_keeps_waiting():
    h = Harness([])

    for _ in range(5):
        h.tick()

    assert h.factory.processes == [] and h.official.calls == 5 and h.log == []


def test_one_assignment_starts_one_worker_with_primitive_input():
    h = Harness([a("seat-1")])

    h.tick()

    (process,) = h.factory.processes
    assert process.target is _tournament_process_entry and process.daemon is True
    assert process.args == (TournamentWorkerInput("seat-1", "match-seat-1", "pokemon_vgc_doubles_draft", SPEC,
                                                  h.state.execution_id),)
    assert h.log == ["Match assigned: pokemon_vgc_doubles_draft", "Starting match..."]


def test_a_tournament_game_logs_its_tournament_round_opponents_and_deadline():
    h = Harness([tournament_game()])

    h.tick()

    assert h.log == [
        "Match assigned: werewolf (tournament)",
        "  Tournament: Fall Cup, Swiss round 1 of 3",
        "  Opponents: Alpha, Beta",
        "  Connect by: 2026-10-16 15:03:40 UTC (3m 40s left)",
        "Starting match...",
    ]


def test_a_testing_game_is_labelled_as_testing():
    h = Harness([a("seat-1", game_type="werewolf", context="testing")])

    h.tick()

    assert h.log == ["Match assigned: werewolf (Testing)", "Starting match..."]


def test_the_tournament_id_is_handed_to_the_worker():
    h = Harness([tournament_game("seat-1"), a("seat-2", context="testing")])

    h.tick()

    assert [p.args[0].tournament_id for p in h.factory.processes] == ["t-1", None]
    assert "t-1" in repr(h.factory.processes[0].args[0])


def test_describe_assignment_without_the_optional_fields_is_one_line():
    assert describe_assignment(a("seat-1", game_type=None), now=NOW) == ["Match assigned: unknown game"]


def test_describe_assignment_round_without_tournament_name():
    lines = describe_assignment(a("seat-1", round_label="Final"), now=NOW)

    assert lines[1:] == ["  Round: Final"]


def test_describe_assignment_tournament_name_without_round():
    lines = describe_assignment(a("seat-1", tournament_name="Fall Cup"), now=NOW)

    assert lines[1:] == ["  Tournament: Fall Cup"]


@pytest.mark.parametrize(
    "deadline, expected",
    [
        ("2026-10-16T15:00:45Z", "  Connect by: 2026-10-16 15:00:45 UTC (45s left)"),
        ("2026-10-16T16:30:00+00:00", "  Connect by: 2026-10-16 16:30:00 UTC (1h 30m left)"),
        ("2026-10-16T08:04:00-07:00", "  Connect by: 2026-10-16 15:04:00 UTC (4m 00s left)"),
        ("2026-10-16T15:04:00", "  Connect by: 2026-10-16 15:04:00 UTC (4m 00s left)"),
        ("2026-10-16T14:59:00Z", "  Connect by: 2026-10-16 14:59:00 UTC (deadline passed)"),
        ("2026-10-16T15:00:00Z", "  Connect by: 2026-10-16 15:00:00 UTC (deadline passed)"),
        ("soon", "  Connect by: soon"),
    ],
)
def test_describe_assignment_connect_deadline(deadline, expected):
    lines = describe_assignment(a("seat-1", connect_deadline_at=deadline), now=NOW)

    assert lines[-1] == expected


def test_describe_assignment_uses_the_wall_clock_by_default():
    deadline = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    (line,) = describe_assignment(a("seat-1", connect_deadline_at=deadline))[1:]

    assert line.endswith("left)") and ("9m" in line or "10m" in line)


def test_describe_assignment_cleans_server_text_before_printing():
    lines = describe_assignment(
        tournament_game(
            tournament_name="Cup\x1b[31mRED\x1b[0m",
            opponents=("Evil\x1b]0;title\x07Bot", "\n\t", "L" * 200),
        ),
        now=NOW,
    )

    joined = "\n".join(lines)
    assert "\x1b" not in joined and "\x07" not in joined
    assert lines[1] == "  Tournament: Cup[31mRED[0m, Swiss round 1 of 3"
    opponents = lines[2].removeprefix("  Opponents: ").split(", ")
    assert opponents[0] == "Evil]0;titleBot"
    assert len(opponents) == 2 and opponents[1] == "L" * 77 + "..."


def test_describe_assignment_never_prints_ids():
    joined = "\n".join(describe_assignment(tournament_game(), now=NOW))

    assert "seat-1" not in joined and "match-seat-1" not in joined and "t-1" not in joined


def test_multiple_simultaneous_assignments_each_get_a_worker():
    h = Harness([a("seat-1"), a("seat-2", game_type="werewolf"), a("seat-3", match_id="match-1")])

    h.tick()

    assert h.started() == ["seat-1", "seat-2", "seat-3"]
    assert h.state.registry.pids().keys() == {"seat-1", "seat-2", "seat-3"}


def test_one_execution_id_per_runtime_shared_by_all_its_workers():
    h = Harness([a("seat-1"), a("seat-2", game_type="werewolf"), a("seat-3")])

    h.tick()

    ids = {p.args[0].execution_id for p in h.factory.processes}
    assert ids == {h.state.execution_id}
    assert 16 <= len(h.state.execution_id) <= 256
    assert h.state.execution_id not in repr(h.factory.processes[0].args[0])  # kept out of logs


def test_each_runtime_gets_a_different_execution_id():
    assert len({TournamentState().execution_id for _ in range(20)}) == 20


def test_execution_id_stays_the_same_across_ticks_and_restarted_workers():
    h = Harness([a("seat-1")], cooldown=1.0)
    h.tick()
    first = h.state.execution_id
    h.factory.processes[0].finish(EXIT_UNEXPECTED)

    h.tick(advance=5.0)
    h.tick(advance=5.0)

    assert h.state.execution_id == first
    assert [p.args[0].execution_id for p in h.factory.processes] == [first, first]


def test_no_duplicate_worker_for_a_seat_already_being_played():
    h = Harness([a("seat-1")])

    for _ in range(4):
        h.tick()

    assert h.started() == ["seat-1"]


def test_finished_match_is_reaped_logged_and_not_restarted_while_it_lingers():
    h = Harness([a("seat-1")], [a("seat-1")], [])
    h.tick()
    h.factory.processes[0].finish(EXIT_SUCCESS)

    h.tick()  # reaped; the seat is still listed for a moment
    h.tick()

    assert h.log[-2:] == ["Match finished.", WAITING_MESSAGE]
    assert h.started() == ["seat-1"]


def test_a_new_assignment_after_a_finish_starts_immediately():
    h = Harness([a("seat-1")], [a("seat-2")])
    h.tick()
    h.factory.processes[0].finish(EXIT_SUCCESS)

    h.tick()

    assert h.started() == ["seat-1", "seat-2"]


def test_failed_worker_is_retried_after_cooldown_reconnect_path():
    h = Harness([a("seat-1")], cooldown=30.0)
    h.tick()
    h.factory.processes[0].finish(EXIT_UNEXPECTED)

    h.tick(advance=1.0)   # reaped -> cooldown
    h.tick(advance=10.0)  # still cooling down
    assert h.started() == ["seat-1"]
    assert "Retrying in 30s" in h.log[2]

    h.tick(advance=30.0)  # cooldown over, still assigned -> new worker (re-grant)
    assert h.started() == ["seat-1", "seat-1"]


def test_a_seat_that_keeps_failing_is_retried_less_and_less_often():
    # A game the server lost answers SESSION_NOT_FOUND until the platform closes it.
    h = Harness([a("seat-1")], cooldown=60.0)
    waits = []
    for _ in range(6):
        h.tick()
        h.factory.processes[-1].finish(EXIT_MATCH_FAILURE)
        h.tick(advance=0.0)  # reaped
        waits.append(h.state.failed_until["seat-1"] - h.clock["t"])
        h.clock["t"] = h.state.failed_until["seat-1"]  # wait it out; the next tick starts a new worker

    assert waits == [60.0, 120.0, 240.0, 480.0, SEAT_FAILURE_BACKOFF_MAX_SECONDS, SEAT_FAILURE_BACKOFF_MAX_SECONDS]
    assert "Retrying in 120s" in "\n".join(h.log)


def test_a_finished_match_resets_the_seat_backoff_and_a_seat_that_left_is_forgotten():
    h = Harness([a("seat-1")], [a("seat-1")], [a("seat-1")], [a("seat-1")], [], cooldown=10.0)
    h.tick()
    h.factory.processes[-1].finish(EXIT_MATCH_FAILURE)
    h.tick(advance=0.0)
    assert h.state.seat_failures == {"seat-1": 1}
    h.clock["t"] = h.state.failed_until["seat-1"]
    h.tick()  # retried
    h.factory.processes[-1].finish(EXIT_SUCCESS)
    h.tick(advance=0.0)
    assert h.state.seat_failures == {}
    assert h.state.failed_until["seat-1"] - h.clock["t"] == 10.0

    h2 = Harness([a("seat-1")], [], cooldown=10.0)
    h2.tick()
    h2.factory.processes[-1].finish(EXIT_MATCH_FAILURE)
    h2.tick(advance=0.0)  # reaped; the seat is no longer listed
    assert h2.state.seat_failures == {}


def test_disappeared_assignment_stops_its_worker_after_grace_polls():
    h = Harness([a("seat-1"), a("seat-2")], [a("seat-2")])
    h.tick()
    seat_1 = h.factory.processes[0]

    for _ in range(MISSING_POLLS_BEFORE_STOP - 1):
        h.tick()
    assert not seat_1.terminated

    h.tick()
    assert seat_1.terminated
    assert h.state.registry.pids().keys() == {"seat-2"}
    assert "Match is no longer assigned; stopped its worker." in h.log


def test_reappearing_seat_resets_the_missing_count():
    h = Harness([a("seat-1")], [], [a("seat-1")], [], [a("seat-1")])

    for _ in range(5):
        h.tick()

    assert not h.factory.processes[0].terminated


def test_transient_discovery_failure_is_logged_once_and_recovers():
    h = Harness(PlatformError("boom", status_code=503), PlatformError("boom", status_code=503), [a("seat-1")])

    h.tick()
    h.tick()
    h.tick()

    assert sum("Could not check tournament assignments" in line for line in h.log) == 1
    assert "Assignment discovery recovered." in h.log
    assert h.started() == ["seat-1"]


def test_authentication_failure_propagates():
    h = Harness(OfficialAgentError("The Official Agent Key was not accepted.", status_code=401))

    with pytest.raises(AuthenticationError):
        h.tick()


# -- what the runtime says about a game's process --------------------------------------------

STOPPED_WITH_ERROR = ("Your agent's process for this match stopped with an error (exit code 1); "
                      "the match continues. Retrying in 60s if it is still assigned.")


def players(count, *, mine=None, game_type="werewolf", context="testing", match_id="m-1"):
    """``mine`` (default: all ``count``) assignments for one match of ``count`` players."""
    return [OfficialAssignment(match_id=match_id, seat_id=f"{match_id}-p{i}", game_type=game_type,
                               seat_position=i, seat_count=count, match_status="in_progress",
                               seat_status="pending", context=context)
            for i in range(count if mine is None else mine)]


def test_an_agent_error_says_the_process_stopped_and_the_match_continues():
    # Was: "Match ended with an error (exit code 1); ..." then "Waiting for
    # your next game..." while the match was still being played.
    h = Harness([a("seat-1")])
    h.tick()
    h.factory.processes[0].finish(EXIT_MATCH_FAILURE)

    h.tick()

    assert h.log[2:] == [STOPPED_WITH_ERROR]
    assert not any("Match ended" in line for line in h.log)


def test_a_match_that_ends_while_its_process_is_stopped_is_said_once():
    # Was: nothing at all when the match ended (e.g. the placeholder in Red Alert).
    h = Harness(players(2, game_type="red_alert"), players(2, game_type="red_alert"), [])
    h.tick()
    for process in h.factory.processes:
        process.finish(EXIT_MATCH_FAILURE)
    h.tick()  # both stopped; the match goes on (still listed)
    assert h.log[-1] == STOPPED_WITH_ERROR and h.log.count(STOPPED_WITH_ERROR) == 1

    h.tick()  # the match is over: no longer listed

    assert h.log[-2:] == ["Your red_alert match (Testing) has ended while your agent's process for it was stopped.",
                          WAITING_MESSAGE]
    h.tick()
    assert h.log.count(WAITING_MESSAGE) == 1


def test_a_game_never_reached_after_its_connect_window_is_not_called_an_error():
    h = Harness([a("seat-1")])
    h.tick()
    h.clock["t"] += MIN_CONNECT_WINDOW_SECONDS
    h.factory.processes[0].finish(EXIT_NOT_CONNECTED)

    h.tick()

    assert h.log[-1] == "Couldn't reach that game (a temporary problem); retrying in 60s if it is still assigned."


def test_waiting_is_said_once_when_one_match_finishes_and_another_ended_without_its_process():
    h = Harness([a("seat-1"), a("seat-2")], [a("seat-1"), a("seat-2")], [a("seat-1")])
    h.tick()
    h.factory.processes[1].finish(EXIT_MATCH_FAILURE)
    h.tick()
    h.factory.processes[0].finish(EXIT_SUCCESS)

    h.tick()  # seat-1's match finished; seat-2's match ended while its process was stopped

    assert h.log[-3:] == ["Match finished.", WAITING_MESSAGE,
                          "Your pokemon_vgc_doubles_draft match has ended while your agent's process for it was stopped."]


def test_a_stopped_process_that_was_started_again_and_finished_says_nothing_more():
    h = Harness([a("seat-1")], [a("seat-1")], [a("seat-1")], [], cooldown=10.0)
    h.tick()
    h.factory.processes[0].finish(EXIT_MATCH_FAILURE)
    h.tick()
    h.tick(advance=10.0)  # started again
    h.factory.processes[1].finish(EXIT_SUCCESS)

    h.tick()
    h.tick()

    assert h.log[-2:] == ["Match finished.", WAITING_MESSAGE]
    assert not any("has ended while" in line for line in h.log)


def test_one_player_that_stopped_in_a_self_play_match_that_finished_says_nothing_more():
    h = Harness(players(3), players(3), [])
    h.tick()
    h.factory.processes[0].finish(EXIT_MATCH_FAILURE)
    h.tick()
    for process in h.factory.processes[1:3]:
        process.finish(EXIT_SUCCESS)

    h.tick()
    h.tick()

    assert h.log[-2:] == ["Match finished.", WAITING_MESSAGE]
    assert not any("has ended while" in line for line in h.log)


def test_a_self_play_pickup_is_one_line_not_one_per_player():
    h = Harness(players(7))

    h.tick()

    assert len(h.factory.processes) == 7
    assert h.log == ["Match assigned: werewolf (Testing), self-play: your agent plays all 7 players",
                     "Starting match..."]


def test_a_pickup_of_some_of_the_players_says_how_many():
    h = Harness(players(4, mine=2))

    h.tick()

    assert h.log == ["Match assigned: werewolf (Testing), your agent plays 2 of the 4 players", "Starting match..."]


def test_a_self_play_finish_is_one_line_not_one_per_player():
    # Was: 7 x ("Match finished.", "Waiting for your next game...").
    h = Harness(players(7), players(7), [])
    h.tick()
    for process in h.factory.processes[:3]:
        process.finish(EXIT_SUCCESS)
    h.tick()  # 3 of 7 done: the match isn't over for this runtime yet
    assert "Match finished." not in h.log
    for process in h.factory.processes[3:]:
        process.finish(EXIT_SUCCESS)

    h.tick()

    assert h.log[2:] == ["Match finished.", WAITING_MESSAGE]


def test_players_held_by_another_copy_are_said_once_per_match():
    h = Harness(players(4))
    h.tick()
    for process in h.factory.processes:
        process.finish(EXIT_SEAT_BUSY)

    h.tick()

    busy = [line for line in h.log if line.startswith("Another runtime is playing this match")]
    assert len(busy) == 1 and h.log[-1] == WAITING_MESSAGE


def test_a_self_play_match_no_longer_assigned_says_so_once():
    h = Harness(players(3), [], [])
    h.tick()

    for _ in range(MISSING_POLLS_BEFORE_STOP):
        h.tick()

    assert all(p.terminated for p in h.factory.processes)
    assert h.log.count("Match is no longer assigned; stopped its worker.") == 1
    assert h.log[-1] == WAITING_MESSAGE and h.log.count(WAITING_MESSAGE) == 1


# -- signing in again during a run (the agent session lasts about an hour) ------------------

MINT_FAILURE = OfficialAgentError(
    "temporary", status_code=401, error_code="invalid_official_agent_key",
    detail="Failed to mint agent session: Request rate limit reached",
)
TEMPORARY_SIGN_IN_FAILURES = [
    pytest.param(OfficialAgentError("slow down", status_code=429, error_code="rate_limited"), id="429"),
    pytest.param(OfficialAgentError("boom", status_code=500, error_code="internal_error"), id="500"),
    pytest.param(OfficialAgentError("gateway", status_code=503), id="503-non-json"),
    pytest.param(OfficialAgentError("gateway", status_code=504), id="504"),
    pytest.param(OfficialAgentError("busy", status_code=503, error_code="agent_session_unavailable"), id="503-unavailable"),
    pytest.param(MINT_FAILURE, id="401-mint-failure"),
    pytest.param(AuthenticationError("Invalid or expired agent session token", status_code=401,
                                     error_code="invalid_agent_session"), id="fresh-session-still-401"),
    pytest.param(OfficialAgentError("no token", status_code=200), id="200-without-token"),
]


@pytest.mark.parametrize("error", TEMPORARY_SIGN_IN_FAILURES)
def test_a_temporary_sign_in_failure_keeps_running_games_and_the_runtime(error):
    h = Harness([a("seat-1")], error, error, [a("seat-1")])
    h.tick()
    worker = h.factory.processes[0]

    h.tick()                 # fails: no exception, the worker keeps playing
    h.tick(advance=120.0)    # fails again after the pause
    h.tick(advance=120.0)    # recovered

    assert not worker.terminated and worker.is_alive()
    assert h.state.registry.pids().keys() == {"seat-1"}
    assert h.started() == ["seat-1"]
    assert sum("Could not renew your agent session" in line for line in h.log) == 1
    assert h.log[-1] == "Assignment discovery recovered."
    assert h.state.auth_failures == 0 and h.state.discovery_retry_at is None


def test_a_sign_in_failure_never_counts_toward_stopping_a_worker():
    error = OfficialAgentError("boom", status_code=502)
    h = Harness([a("seat-1")], *([error] * (MISSING_POLLS_BEFORE_STOP + 3)), [a("seat-1")])
    h.tick()

    for _ in range(MISSING_POLLS_BEFORE_STOP + 4):
        h.tick(advance=120.0)

    assert not h.factory.processes[0].terminated
    assert h.state.missing_polls == {}


def test_after_too_many_attempts_it_waits_a_full_minute_before_trying_again():
    h = Harness(OfficialAgentError("slow down", status_code=429, error_code="rate_limited"), [])

    h.tick(advance=1.0)       # t=1: 429
    for _ in range(5):
        h.tick(advance=10.0)  # t=11..51: still waiting
    assert h.official.calls == 1

    h.tick(advance=10.0)      # t=61: a minute later
    assert h.official.calls == 2


def test_server_errors_back_off_10_20_40_then_60_seconds():
    h = Harness(OfficialAgentError("boom", status_code=500, error_code="internal_error"))
    attempts = []

    for _ in range(200):
        before = h.official.calls
        h.tick(advance=1.0)
        if h.official.calls > before:
            attempts.append(h.clock["t"])

    gaps = [later - earlier for earlier, later in zip(attempts, attempts[1:])]
    assert gaps[:5] == [10.0, 20.0, 40.0, 60.0, 60.0]


def test_the_backoff_starts_over_after_a_successful_listing():
    error = OfficialAgentError("boom", status_code=500)
    h = Harness(error, error, [], error, [])

    h.tick(advance=1.0)    # fail (wait 10)
    h.tick(advance=10.0)   # fail (wait 20)
    h.tick(advance=20.0)   # ok
    h.tick(advance=1.0)    # fail again: back to a 10 s wait
    assert h.state.discovery_retry_at == h.clock["t"] + 10.0


def test_finished_workers_are_still_reaped_while_sign_in_is_failing():
    h = Harness([a("seat-1")], OfficialAgentError("slow down", status_code=429, error_code="rate_limited"))
    h.tick()
    h.tick()  # 429: waiting a minute
    h.factory.processes[0].finish(EXIT_SUCCESS)

    h.tick(advance=5.0)

    assert h.log[-2:] == ["Match finished.", WAITING_MESSAGE]
    assert len(h.state.registry) == 0


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OfficialAgentError("not accepted", status_code=401, error_code="invalid_official_agent_key",
                                        detail="Invalid official agent key"), id="wrong-or-revoked-key"),
        pytest.param(OfficialAgentError("oracle", status_code=401, error_code="invalid_official_agent_key",
                                        detail="This agent is Oracle Hosted; ..."), id="oracle-hosted"),
    ],
)
def test_a_refused_key_still_stops_the_runtime(error):
    h = Harness([a("seat-1")], error)
    h.tick()

    with pytest.raises(OfficialAgentError):
        h.tick()


def test_forever_survives_temporary_sign_in_failures_without_stopping_workers():
    official = FakeOfficial([a("seat-1")], MINT_FAILURE, OfficialAgentError("x", status_code=503), [a("seat-1")])
    factory = FakeProcessFactory()
    clock = {"t": 0.0}
    log: list[str] = []

    def sleep(seconds):
        clock["t"] += 30.0

    run_tournament_forever(official, agent_spec=SPEC, sleep=sleep, now=lambda: clock["t"],
                           process_factory=factory, log=log.append, max_iterations=8)

    assert len(factory.processes) == 1
    assert official.calls >= 4
    assert "Assignment discovery recovered." in log
    assert not any("Stopping" in line for line in log[:-1])


# -- the same, through the real client and a mock control plane ------------------------------


class RefreshBackend(OfficialBackend):
    """The control plane from test_official, where the next sign-in answers
    ``refresh`` (one response, used while set) instead of a session.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.refresh: httpx.Response | None = None
        self.reject_new_sessions = False

    def __call__(self, request):
        if request.url.path == AUTHENTICATE_PATH and self.refresh is not None:
            self.auth_bodies.append({})
            return self.refresh
        response = super().__call__(request)
        if request.url.path == AUTHENTICATE_PATH and self.reject_new_sessions:
            self.valid.clear()
        return response


REFRESH_ANSWERS = [
    pytest.param(httpx.Response(500, json={"error": "internal_error", "detail": "Failed to authenticate"}), id="500"),
    pytest.param(httpx.Response(429, json={"error": "rate_limited", "detail": "Too many requests."}), id="429"),
    pytest.param(httpx.Response(503, text="<html>Service Unavailable</html>"), id="503-html"),
    pytest.param(httpx.Response(504, json={"message": "Endpoint request timed out"}), id="504"),
    pytest.param(httpx.Response(401, json={"error": "invalid_official_agent_key",
                                           "detail": "Failed to mint agent session: Request rate limit reached"}),
                 id="401-mint-failure"),
    pytest.param(httpx.Response(503, json={"error": "agent_session_unavailable", "detail": "try again"}),
                 id="503-agent-session-unavailable"),
]


def _real_harness(backend):
    h = Harness([])
    h.official = OfficialAgentClient(CONTROL, KEY, load_env_file=False, transport=httpx.MockTransport(backend))
    return h


@pytest.mark.parametrize("answer", REFRESH_ANSWERS)
def test_real_client_session_refresh_failure_keeps_games_running(answer):
    backend = RefreshBackend(assignments=[official_assignment("seat-1")])
    h = _real_harness(backend)
    h.tick()
    worker = h.factory.processes[0]
    backend.valid.clear()   # the hour-long session expires
    backend.refresh = answer

    h.tick()                # GET 401 -> sign in again -> temporary failure

    assert not worker.terminated and h.state.registry.pids().keys() == {"seat-1"}
    assert any("Could not renew your agent session" in line for line in h.log)
    assert not any("Rotate key" in line for line in h.log)

    backend.refresh = None  # the platform recovers
    h.tick(advance=120.0)
    assert h.log[-1] == "Assignment discovery recovered."
    assert not worker.terminated and h.started() == ["seat-1"]


def test_real_client_fresh_session_still_rejected_keeps_games_running():
    backend = RefreshBackend(assignments=[official_assignment("seat-1")])
    h = _real_harness(backend)
    h.tick()
    worker = h.factory.processes[0]
    backend.valid.clear()
    backend.reject_new_sessions = True  # e.g. the sign-in service is briefly down

    h.tick()

    assert not worker.terminated
    assert any("Could not renew your agent session" in line for line in h.log)

    backend.reject_new_sessions = False
    h.tick(advance=120.0)
    assert h.log[-1] == "Assignment discovery recovered."


def test_real_client_revoked_key_still_stops_the_runtime():
    backend = RefreshBackend(assignments=[official_assignment("seat-1")])
    h = _real_harness(backend)
    h.tick()
    backend.valid.clear()
    backend.key_valid = False

    with pytest.raises(OfficialAgentError) as exc_info:
        h.tick()
    assert exc_info.value.error_code == "invalid_official_agent_key"


# -- which kinds of game a runtime plays (--tournament / --match) ----------------------------

MATCH_ONLY = frozenset({TESTING})
TOURNAMENT_ONLY = frozenset({TOURNAMENT})
FALL_CUP_WARNING = ("You have a tournament game waiting (Fall Cup, Swiss round 1 of 3): run with --tournament "
                    "to play it — it counts as a loss if your agent doesn't connect within the window.")


def match_game(seat_id="seat-t", **overrides):
    return a(seat_id, game_type="werewolf", context="testing", **overrides)


@pytest.mark.parametrize(
    "context, kind",
    [("testing", TESTING), ("Testing", TESTING), (" testing ", TESTING), ("tournament", TOURNAMENT),
     (None, TOURNAMENT), ("", TOURNAMENT), ("exhibition", TOURNAMENT)],
)
def test_assignment_kind_missing_or_unknown_context_is_a_tournament_game(context, kind):
    assert assignment_kind(a("seat-1", context=context)) == kind


def test_match_only_plays_test_matches_and_skips_tournament_games():
    h = Harness([tournament_game("seat-1"), match_game("seat-2")], kinds=MATCH_ONLY)

    h.tick()

    assert h.started() == ["seat-2"]
    assert h.log == [FALL_CUP_WARNING, "Match assigned: werewolf (Testing)", "Starting match..."]


def test_tournament_only_plays_tournament_games_and_notes_test_matches():
    h = Harness([match_game("seat-2"), tournament_game("seat-1")], kinds=TOURNAMENT_ONLY)

    h.tick()

    assert h.started() == ["seat-1"]
    assert h.log[0] == TEST_MATCH_WAITING_NOTE == "Test match waiting: run with --match to play it"
    assert h.log[1] == "Match assigned: werewolf (tournament)"


def test_an_assignment_without_a_context_is_played_as_a_tournament_game():
    assert a("seat-1").context is None
    tournament = Harness([a("seat-1")], kinds=TOURNAMENT_ONLY)
    match = Harness([a("seat-1")], kinds=MATCH_ONLY)

    tournament.tick()
    match.tick()

    assert tournament.started() == ["seat-1"]
    assert match.started() == []
    assert match.log == ["You have a tournament game waiting (pokemon_vgc_doubles_draft): run with --tournament "
                         "to play it — it counts as a loss if your agent doesn't connect within the window."]


def test_both_flags_play_both_kinds_in_one_runtime_without_notes():
    h = Harness([tournament_game("seat-1"), match_game("seat-2"), a("seat-3")],
                kinds=frozenset({TESTING, TOURNAMENT}))

    h.tick()

    assert h.started() == ["seat-1", "seat-2", "seat-3"]
    assert FALL_CUP_WARNING not in h.log and TEST_MATCH_WAITING_NOTE not in h.log


def test_the_tournament_warning_is_shown_once_per_game():
    second = tournament_game("seat-9", match_id="match-9", tournament_name="Winter Cup", round_label="Final")
    h = Harness([tournament_game("seat-1")], [tournament_game("seat-1"), second], kinds=MATCH_ONLY)

    for _ in range(5):
        h.tick()

    assert h.log == [
        FALL_CUP_WARNING,
        "You have a tournament game waiting (Winter Cup, Final): run with --tournament to play it — "
        "it counts as a loss if your agent doesn't connect within the window.",
    ]
    assert h.started() == []


def test_the_test_match_note_is_shown_once_per_game_even_with_several_of_its_seats():
    self_play = [match_game("seat-a", match_id="m-1"), match_game("seat-b", match_id="m-1")]
    h = Harness(self_play, self_play + [match_game("seat-c", match_id="m-2")], kinds=TOURNAMENT_ONLY)

    for _ in range(4):
        h.tick()

    assert h.log == [TEST_MATCH_WAITING_NOTE, TEST_MATCH_WAITING_NOTE]
    assert h.started() == []


@pytest.mark.parametrize(
    "fields, detail",
    [
        (dict(tournament_name="Fall Cup", round_label="Swiss round 1 of 3"), " (Fall Cup, Swiss round 1 of 3)"),
        (dict(tournament_name="Fall Cup"), " (Fall Cup)"),
        (dict(round_label="Final"), " (Final)"),
        (dict(), " (werewolf)"),
        (dict(game_type=None), ""),
        (dict(tournament_name="Bad\x1b[31mCup"), " (Bad[31mCup)"),
    ],
)
def test_tournament_waiting_warning_names_what_the_server_sent(fields, detail):
    fields = {"game_type": "werewolf", **fields}

    assert tournament_waiting_warning(a("seat-1", context="tournament", **fields)) == (
        f"You have a tournament game waiting{detail}: run with --tournament to play it — "
        "it counts as a loss if your agent doesn't connect within the window."
    )


def test_a_running_game_is_not_stopped_when_its_context_changes_to_a_kind_not_played():
    # Started as a tournament game (no context: an older backend), then the
    # backend starts labelling the same seat "testing".
    h = Harness([a("seat-1")], [match_game("seat-1", match_id="match-seat-1")], kinds=TOURNAMENT_ONLY)

    for _ in range(MISSING_POLLS_BEFORE_STOP + 3):
        h.tick()

    assert not h.factory.processes[0].terminated
    assert h.state.registry.is_active("seat-1")
    assert TEST_MATCH_WAITING_NOTE not in h.log and h.started() == ["seat-1"]


def test_a_running_game_keeps_playing_while_games_of_the_other_kind_are_skipped():
    h = Harness([match_game("seat-1")],
                [match_game("seat-1"), tournament_game("seat-2")], kinds=MATCH_ONLY)

    for _ in range(MISSING_POLLS_BEFORE_STOP + 3):
        h.tick()

    assert h.started() == ["seat-1"] and not h.factory.processes[0].terminated
    assert h.log.count(FALL_CUP_WARNING) == 1


def test_a_skipped_game_is_never_started_after_a_played_one_finishes():
    h = Harness([match_game("seat-1"), tournament_game("seat-2")], kinds=MATCH_ONLY)
    h.tick()
    h.factory.processes[0].finish(EXIT_SUCCESS)

    for _ in range(3):
        h.tick(advance=100.0)  # past every cooldown

    assert h.started() == ["seat-1", "seat-1"]  # the test match again (still listed), never the tournament game
    assert h.log.count(FALL_CUP_WARNING) == 1


def test_forever_hands_its_kinds_to_every_tick():
    official = FakeOfficial([tournament_game("seat-1"), match_game("seat-2")])
    factory = FakeProcessFactory()
    log: list[str] = []

    run_tournament_forever(official, agent_spec=SPEC, kinds=MATCH_ONLY, sleep=lambda s: None,
                           process_factory=factory, log=log.append, max_iterations=3)

    assert [p.args[0].seat_id for p in factory.processes] == ["seat-2"]
    assert log.count(FALL_CUP_WARNING) == 1


def test_forever_plays_both_kinds_by_default():
    official = FakeOfficial([tournament_game("seat-1"), match_game("seat-2")])
    factory = FakeProcessFactory()

    run_tournament_forever(official, agent_spec=SPEC, sleep=lambda s: None, process_factory=factory,
                           log=lambda line: None, max_iterations=1)

    assert [p.args[0].seat_id for p in factory.processes] == ["seat-1", "seat-2"]


def test_forever_keeps_polling_with_no_assignments_and_stops_workers_on_ctrl_c():
    official = FakeOfficial([a("seat-1"), a("seat-2")])
    factory = FakeProcessFactory()
    sleeps = []
    log: list[str] = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_tournament_forever(official, agent_spec=SPEC, sleep=sleep, process_factory=factory, log=log.append)

    assert official.calls == 3
    assert all(p.terminated for p in factory.processes)
    assert log[-1] == "Stopping 2 active match worker(s)..."


def test_forever_never_exits_on_its_own_when_nothing_is_assigned():
    official = FakeOfficial([])

    run_tournament_forever(official, agent_spec=SPEC, sleep=lambda s: None, max_iterations=25)

    assert official.calls == 25


# -- worker ----------------------------------------------------------------------------------


def _grant(**overrides):
    data = {"access_token": "seat-jwt", "agent_id": "synthetic-1", "gameapi_server_url": "https://gameapi.example.test",
            "game_session_id": "game-1", "match_id": "match-1", "seat_id": "seat-1", "seat_position": 1,
            "seat_count": 2, "game_type": "pokemon_vgc_doubles_draft", "match_status": "in_progress"}
    data.update(overrides)
    return SeatGrant.from_dict(data)


class FakeWorkerOfficial:
    instances: list["FakeWorkerOfficial"] = []

    def __init__(self, grant_result=None):
        self.control_url = "https://control.example.test"
        self.grant_result = grant_result if grant_result is not None else _grant()
        self.grants: list[str] = []
        self.execution_ids: list[str] = []
        self.renewals: list[tuple[str, str]] = []
        self.session_tokens: list[str] = []
        self.closed = False
        FakeWorkerOfficial.instances.append(self)

    def grant(self, seat_id, execution_id):
        self.execution_ids.append(execution_id)
        self.grants.append(seat_id)
        result = self.grant_result
        if isinstance(result, list):  # one answer per call; the last one repeats
            result = result.pop(0) if len(result) > 1 else result[0]
        if isinstance(result, Exception):
            raise result
        return result

    def use_session_token(self, token):
        self.session_tokens.append(token)

    def renew_lease(self, seat_id, execution_id):
        self.renewals.append((seat_id, execution_id))
        return {"seat_id": seat_id, "execution_lease_expires_at": "x"}

    def close(self):
        self.closed = True


def _worker(spec=SPEC, grant_result=None, run_game_fn=None):
    runs = []

    def fake_run_game(game, context, contestant):
        runs.append((game, context, contestant))
        return types.SimpleNamespace(termination_reason="normal", returns={"synthetic-1": 1.0})

    officials = []

    def factory():
        officials.append(FakeWorkerOfficial(grant_result))
        return officials[-1]

    code = run_tournament_worker(TournamentWorkerInput("seat-1", "match-1", "pokemon_vgc_doubles_draft", spec, EXEC),
                                 official_factory=factory, run_game_fn=run_game_fn or fake_run_game)
    return code, runs, officials


def _write_module(tmp_path, monkeypatch, name, body):
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)


def test_worker_grants_the_seat_and_plays_it_through_run_game(capsys):
    code, runs, officials = _worker()

    assert code == EXIT_SUCCESS
    (game, context, contestant), = runs
    assert isinstance(game, MCPGameSession)
    assert (game.session_id, game.game_server_url) == ("game-1", "https://gameapi.example.test")
    assert game._client.auth.seat_id == "seat-1"
    assert (context.session_id, context.agent_id, context.seat_position, context.game_type) == (
        "game-1", "synthetic-1", 1, "pokemon_vgc_doubles_draft")
    assert officials[0].grants == ["seat-1"] and officials[0].closed
    out = capsys.readouterr().out
    assert "finished: your agent (player 2) won (termination_reason=normal, score=1.0)" in out
    assert "seat-jwt" not in out


def _finish_line(capsys, termination_reason="completed", returns=None, final_result=None):
    def fake_run_game(game, context, contestant):
        return types.SimpleNamespace(termination_reason=termination_reason, returns=returns,
                                     final_result=final_result)

    code = run_tournament_worker(TournamentWorkerInput("seat-1", "match-1", "werewolf", SPEC, EXEC),
                                 official_factory=lambda: FakeWorkerOfficial(None), run_game_fn=fake_run_game)
    assert code == EXIT_SUCCESS
    (line,) = [line for line in capsys.readouterr().out.splitlines() if " finished" in line]
    return line.split("] ", 1)[1]


def test_worker_says_won_or_lost_from_the_games_own_result_werewolf(capsys):
    # Werewolf's returns are keyed by player name, not by agent id, so the
    # finished line used to say nothing about the result.
    result = {"returns": {"Player 1": -1.0, "Player 2": 1.0}, "your_return": 1.0, "status": "completed",
              "result": {"winning_team": "wolves", "seats": {"synthetic-1": {"outcome": "win", "team": "wolves"}}}}

    line = _finish_line(capsys, returns=result["returns"], final_result=result)

    assert line == "finished: your agent (player 2) won (termination_reason=completed, score=1.0)"


def test_worker_says_lost_from_the_winner_pokemon_and_red_alert(capsys):
    result = {"returns": {"other": 1.0, "synthetic-1": 0.0}, "your_return": 0.0, "status": "completed",
              "winner_agent_id": "other"}

    line = _finish_line(capsys, returns=result["returns"], final_result=result)

    assert line == "finished: your agent (player 2) lost (termination_reason=completed, score=0.0)"


def test_worker_says_a_draw_from_an_even_score(capsys):
    result = {"returns": {"other": 0.5, "synthetic-1": 0.5}, "your_return": 0.5, "status": "completed",
              "winner_agent_id": None}

    line = _finish_line(capsys, termination_reason="time_limit", returns=result["returns"], final_result=result)

    assert line == "finished: your agent (player 2) drew (termination_reason=time_limit, score=0.5)"


def test_worker_says_no_result_for_a_game_the_server_could_not_finish(capsys):
    result = {"returns": None, "your_return": None, "status": "failed", "winner_agent_id": None,
              "failure_reason": "battle_room_timeout"}

    line = _finish_line(capsys, termination_reason="error", final_result=result)

    assert line == "finished with no result (termination_reason=error)"


def test_worker_without_a_known_result_just_says_finished(capsys):
    line = _finish_line(capsys, termination_reason="normal", returns={})

    assert line == "finished (termination_reason=normal)"


def test_worker_hands_the_tournament_id_to_the_contestant():
    runs = []

    def fake_run_game(game, context, contestant):
        runs.append(context)
        return types.SimpleNamespace(termination_reason="normal", returns={})

    code = run_tournament_worker(
        TournamentWorkerInput("seat-1", "match-1", "werewolf", SPEC, EXEC, tournament_id="t-1"),
        official_factory=lambda: FakeWorkerOfficial(None), run_game_fn=fake_run_game,
    )

    assert code == EXIT_SUCCESS and runs[0].tournament_id == "t-1"


def test_worker_testing_game_has_no_tournament_id():
    _, runs, _ = _worker()

    assert runs[0][1].tournament_id is None


def test_each_worker_builds_its_own_contestant(monkeypatch, tmp_path):
    _write_module(tmp_path, monkeypatch, "tournament_counting_agent", """
        created = []

        class Agent:
            def choose_action(self, state, context):
                return state.legal_actions[0]

        def create_agent():
            created.append(Agent())
            return created[-1]
    """)

    _, runs_a, _ = _worker(spec="tournament_counting_agent")
    _, runs_b, _ = _worker(spec="tournament_counting_agent")

    module = sys.modules["tournament_counting_agent"]
    assert len(module.created) == 2
    assert runs_a[0][2] is module.created[0] and runs_b[0][2] is module.created[1]


def test_worker_agent_override_spec():
    _, runs, _ = _worker(spec="examples.smoke_agent")

    assert runs[0][2].__module__ == "examples.smoke_agent"


def test_worker_restart_requests_a_fresh_grant_each_time():
    _, _, first = _worker()
    _, _, second = _worker()

    assert first[0].grants == ["seat-1"] and second[0].grants == ["seat-1"]
    assert first[0] is not second[0]


def test_worker_exits_cleanly_when_the_assignment_already_ended():
    code, runs, _ = _worker(grant_result=OfficialAgentError("ended", status_code=409, error_code="assignment_not_grantable"))

    assert code == EXIT_SUCCESS and runs == []


def test_worker_factory_error_fails_before_any_grant(monkeypatch, tmp_path):
    _write_module(tmp_path, monkeypatch, "tournament_broken_agent", "def create_agent():\n    raise RuntimeError('no key')\n")

    code, runs, officials = _worker(spec="tournament_broken_agent")

    assert code == EXIT_UNEXPECTED and runs == [] and officials == []


def test_worker_match_failure_exit_code():
    def failing(game, context, contestant):
        raise MCPToolError("The game session was not found.", status_code=None, error_code="SESSION_NOT_FOUND")

    code, _, officials = _worker(run_game_fn=failing)

    assert code == EXIT_MATCH_FAILURE and officials[0].closed


def test_worker_temporary_failure_before_reaching_the_game_is_not_connected():
    def failing(game, context, contestant):
        raise PlatformError("GameAPI error", status_code=500)

    code, _, officials = _worker(run_game_fn=failing)

    assert code == EXIT_NOT_CONNECTED and officials[0].closed


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(MCPToolError("MCP tool 'wait_for_update' failed: ReadTimeout('')", status_code=None,
                                  error_code=None), id="no-answer"),
        pytest.param(MCPToolError("waiting already", status_code=None, error_code="TOO_MANY_WAITS"),
                     id="too-many-waits"),
    ],
)
def test_worker_that_lost_the_connection_after_reaching_the_game_exits_connection_lost(error, capsys):
    # run_game gave up after 90 s of temporary failures. The supervisor needs
    # to know why (a lost connection, not an agent error) to restart it soon.
    def failing(game, context, contestant):
        game.contacted = True  # GameAPI answered at least once
        raise error

    code, _, _ = _worker(run_game_fn=failing)

    assert code == EXIT_CONNECTION_LOST
    assert "lost the connection to the game" in capsys.readouterr().out


@pytest.mark.parametrize("contacted", [False, True])
def test_worker_whose_call_the_server_cant_handle_is_a_match_failure(contacted):
    # An isError answer (arguments the server rejected, or its tool crashed)
    # isn't a connection problem, before or after reaching the game.
    def failing(game, context, contestant):
        game.contacted = contacted
        raise MCPToolError("MCP tool 'play_action' failed at the protocol level: 1 validation error",
                           status_code=None, error_code=None, protocol_error=True)

    code, _, _ = _worker(run_game_fn=failing)

    assert code == EXIT_MATCH_FAILURE


@pytest.mark.parametrize("as_decision_error", [False, True])
def test_worker_whose_move_cant_be_sent_as_json_is_a_match_failure_not_a_lost_connection(as_decision_error, capsys):
    # A move holding, say, a numpy number never leaves the process. That is a
    # bug in the agent's code: "match failed", never "lost the connection".
    unsendable = MCPToolError("MCP tool 'play_action' can't send its arguments as JSON: Unable to serialize",
                              status_code=None, error_code=None, local_error=True)

    def failing(game, context, contestant):
        game.contacted = True
        if as_decision_error:  # as run_game reports it for a move or message
            raise DecisionError("choose_action returned a move that can't be sent") from unsendable
        raise unsendable

    code, _, _ = _worker(run_game_fn=failing)

    out = capsys.readouterr().out
    assert code == EXIT_MATCH_FAILURE
    assert "match failed" in out and "lost the connection" not in out


def test_worker_ctrl_c_exits_quietly():
    def interrupted(game, context, contestant):
        raise KeyboardInterrupt

    code, _, _ = _worker(run_game_fn=interrupted)

    assert code == EXIT_SUCCESS


# -- resilience: hiccups never become no-shows, and a game that was playing is picked up fast --


def _run_ticks(h, ticks, advance, exitcode):
    """Tick ``ticks`` times; every worker started fails at once with
    ``exitcode``. Returns the clock times at which workers started."""
    starts = []
    for _ in range(ticks):
        before = len(h.factory.processes)
        h.tick(advance=advance)
        if len(h.factory.processes) > before:
            starts.append(h.clock["t"])
            h.factory.processes[-1].finish(exitcode)
    return starts


def test_a_game_not_reached_yet_is_tried_again_every_poll_while_its_connect_window_is_open():
    # Was: attempts at 0, 70, 200 and 450 s, so a burst of failures at round
    # start (the platform busy) missed the 220 s window: a no-show loss.
    h = Harness([tournament_game()])  # connect by 15:03:40, 220 s after NOW

    starts = _run_ticks(h, 80, 10.0, EXIT_NOT_CONNECTED)

    in_window = [t for t in starts if t - starts[0] < MIN_CONNECT_WINDOW_SECONDS]
    gaps = [later - earlier for earlier, later in zip(in_window, in_window[1:])]
    assert len(in_window) == MIN_CONNECT_WINDOW_SECONDS / 10.0 and max(gaps) == 10.0
    assert "Couldn't reach that game yet (a temporary problem); trying again now." in h.log
    # Once the window has closed (the platform has decided the game), the
    # normal backoff applies (each gap also includes the poll that notices the exit).
    late = [t for t in starts if t - starts[0] >= MIN_CONNECT_WINDOW_SECONDS]
    assert [later - earlier for earlier, later in zip(late, late[1:])][:2] == [130.0, 250.0]


def test_without_a_deadline_the_connect_window_is_five_minutes_from_when_the_game_was_listed():
    h = Harness([a("seat-1", context="testing")])

    starts = _run_ticks(h, 60, 10.0, EXIT_NOT_CONNECTED)

    assert h.state.connect_by["seat-1"][1] == starts[0] + MIN_CONNECT_WINDOW_SECONDS
    fast = [t for t in starts if t - starts[0] < MIN_CONNECT_WINDOW_SECONDS]
    assert len(fast) == MIN_CONNECT_WINDOW_SECONDS / 10.0
    assert len(starts) - len(fast) <= 3


def test_a_clock_that_makes_the_deadline_look_passed_still_gets_the_fast_retries():
    h = Harness([tournament_game(connect_deadline_at="2026-10-16T14:50:00Z")])  # 10 min "ago"

    starts = _run_ticks(h, 20, 10.0, EXIT_NOT_CONNECTED)

    assert {later - earlier for earlier, later in zip(starts, starts[1:])} == {10.0}


def test_a_later_deadline_keeps_the_window_open_until_then():
    h = Harness([tournament_game(connect_deadline_at="2026-10-16T15:10:00Z")])  # 600 s after NOW

    h.tick(advance=10.0)

    assert h.state.connect_by["seat-1"][1] == 10.0 + 600.0


def test_a_new_game_for_the_same_seat_gets_a_new_connect_window():
    first = tournament_game()
    replay = tournament_game(match_id="match-replay", connect_deadline_at="2026-10-16T15:30:00Z")
    h = Harness([first], [replay])

    h.tick(advance=10.0)
    h.factory.processes[-1].finish(EXIT_SUCCESS)
    h.tick(advance=10.0)

    assert h.state.connect_by["seat-1"] == (("match-replay", "2026-10-16T15:30:00Z"), 20.0 + 1800.0)


def test_a_game_that_was_playing_is_picked_up_again_within_seconds_and_its_count_starts_over():
    # Was: a worker that played five minutes then hit a blip counted as one
    # more failure in a row, so the agent was absent 60, 120, 240, 480, 600 s.
    h = Harness([a("seat-1")])
    absences = []
    h.tick()
    for _ in range(5):
        h.clock["t"] += 300.0  # plays for five minutes, then the worker gives up on a long outage
        h.factory.processes[-1].finish(EXIT_CONNECTION_LOST)
        h.tick(advance=0.0)
        absences.append(h.state.failed_until["seat-1"] - h.clock["t"])
        h.clock["t"] = h.state.failed_until["seat-1"]
        h.tick(advance=0.0)

    assert absences == [RESTART_AFTER_PLAY_SECONDS] * 5
    assert h.state.seat_failures == {"seat-1": 1}
    assert len(h.factory.processes) == 6
    assert "Lost the connection to that game for a while; retrying that game in 10s" in "\n".join(h.log)


def test_a_fast_failure_right_after_a_long_run_continues_the_doubling_from_one():
    h = Harness([a("seat-1")])
    h.tick()
    h.clock["t"] += TRANSIENT_GIVE_UP_SECONDS + WORKER_PLAYED_SECONDS
    h.factory.processes[-1].finish(EXIT_CONNECTION_LOST)
    h.tick(advance=0.0)
    h.clock["t"] = h.state.failed_until["seat-1"]
    h.tick(advance=0.0)  # restarted; this one fails at once
    h.factory.processes[-1].finish(EXIT_MATCH_FAILURE)
    h.tick(advance=0.0)

    assert h.state.seat_failures == {"seat-1": 2}
    assert h.state.failed_until["seat-1"] - h.clock["t"] == 120.0


def _run_workers(h, seconds, run_for, exitcode, poll=10.0):
    """Poll every ``poll`` seconds for ``seconds``; each worker started runs
    ``run_for`` seconds, then exits with ``exitcode``. Returns the clock times
    at which workers started."""
    starts, running = [], None
    while h.clock["t"] < seconds:
        before = len(h.factory.processes)
        h.tick(advance=poll)
        if len(h.factory.processes) > before:
            starts.append(h.clock["t"])
            running = (h.factory.processes[-1], h.clock["t"])
        if running and h.clock["t"] - running[1] >= run_for:
            running[0].finish(exitcode)
            running = None
    return starts


@pytest.mark.parametrize("run_for", [30.0, 60.0, 100.0])
@pytest.mark.parametrize("exitcode", [EXIT_MATCH_FAILURE, EXIT_UNEXPECTED])
def test_an_agent_that_keeps_crashing_during_the_game_is_retried_less_and_less_often(exitcode, run_for):
    # Was: any worker that ran 30 s or more was restarted after 10 s, whatever
    # the reason, so an agent that crashed 30 s into each restart was started
    # (each start one new seat grant) about 72 times an hour.
    h = Harness([a("seat-1")])

    starts = _run_workers(h, 3600.0, run_for, exitcode)

    gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
    assert gaps[:4] == [run_for + wait + 10.0 for wait in (60.0, 120.0, 240.0, 480.0)]
    assert len(starts) <= 9


def test_a_connection_lost_almost_at_once_is_retried_less_and_less_often():
    # The worker reached the game, then every call failed until the in-game
    # retries gave up (about 100 s). Restarting it every 10 s would cost a
    # new seat grant every ~110 s for as long as the game is broken.
    h = Harness([a("seat-1")])

    starts = _run_workers(h, 3600.0, TRANSIENT_GIVE_UP_SECONDS + 10.0, EXIT_CONNECTION_LOST)

    gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
    assert gaps[:3] == [100.0 + wait + 10.0 for wait in (60.0, 120.0, 240.0)]
    assert len(starts) <= 9


def test_after_the_connect_window_a_worker_that_never_gets_in_is_retried_less_and_less_often():
    # A real EXIT_NOT_CONNECTED worker has already retried its grant for a
    # minute or more. Was: after the window closed it still counted as
    # "playing" (it ran 30 s or more), so it was replaced every 80-90 s forever.
    h = Harness([tournament_game()])  # window: MIN_CONNECT_WINDOW_SECONDS from the first listing

    starts = _run_workers(h, 3600.0, 70.0, EXIT_NOT_CONNECTED)

    window_end = starts[0] + MIN_CONNECT_WINDOW_SECONDS
    in_window = [t for t in starts if t < window_end]
    gaps = [later - earlier for earlier, later in zip(in_window, in_window[1:])]
    assert gaps and set(gaps) == {80.0}  # replaced in the poll that notices the exit
    late = [t for t in starts if t >= window_end]
    late_gaps = [later - earlier for earlier, later in zip(late, late[1:])]
    assert late_gaps[:3] == [70.0 + wait + 10.0 for wait in (120.0, 240.0, 480.0)]
    assert len(starts) <= 12


def test_an_agent_error_after_a_long_healthy_run_starts_a_new_count():
    h = Harness([a("seat-1")])
    h.state.seat_failures["seat-1"] = 3  # earlier trouble in this game
    h.tick()
    h.clock["t"] += SEAT_FAILURE_BACKOFF_MAX_SECONDS  # then ten minutes of play
    h.factory.processes[-1].finish(EXIT_MATCH_FAILURE)
    h.tick(advance=0.0)

    assert h.state.seat_failures == {"seat-1": 1}
    assert h.state.failed_until["seat-1"] - h.clock["t"] == 60.0


def test_a_worker_turned_away_by_its_registration_is_retried_every_30s_without_counting():
    h = Harness([a("seat-1")])
    starts = _run_ticks(h, 10, 10.0, EXIT_REGISTRATION_INCOMPLETE)

    gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
    assert gaps and set(gaps) == {REGISTRATION_RETRY_SECONDS + 10.0}  # plus the poll that notices the exit
    assert h.state.seat_failures == {}
    assert "That game can't start until your registration is complete; checking again in 30s." in h.log


RULES_UPDATED = OfficialAgentError(
    "Accept the updated Official Rules on your dashboard (https://platform.altruagent-game.com/tournament/dashboard).",
    status_code=403, error_code="registration_incomplete",
)
RULES_UPDATED.next_step = "rules"
RULES_MESSAGE = ("Accept the updated Official Rules on your dashboard "
                 "(https://platform.altruagent-game.com/tournament/dashboard); I'll keep trying.")


def test_updated_rules_never_stop_the_runtime_or_its_running_games():
    # Was: fatal; the runtime printed "Stopped due to an unrecoverable error"
    # and exited, and every later game was a no-show.
    h = Harness([a("seat-1")], *([RULES_UPDATED] * 4), [a("seat-1"), a("seat-2")])
    h.tick()
    worker = h.factory.processes[0]

    for _ in range(4):
        h.tick(advance=REGISTRATION_RETRY_SECONDS)

    assert not worker.terminated and h.state.registry.is_active("seat-1")
    assert h.log.count(RULES_MESSAGE) == 1  # said once, not every 30 s
    assert h.official.calls == 5
    h.tick(advance=REGISTRATION_RETRY_SECONDS)  # accepted on the dashboard
    assert h.log[-3:] == ["Assignment discovery recovered.", "Match assigned: pokemon_vgc_doubles_draft",
                          "Starting match..."]
    assert h.started() == ["seat-1", "seat-2"]


def test_while_the_registration_is_incomplete_it_retries_slowly_and_reminds_every_few_minutes():
    h = Harness(RULES_UPDATED)
    attempts = []
    for _ in range(700):
        before = h.official.calls
        h.tick(advance=1.0)
        if h.official.calls > before:
            attempts.append(h.clock["t"])

    gaps = {later - earlier for earlier, later in zip(attempts, attempts[1:])}
    assert gaps == {REGISTRATION_RETRY_SECONDS}
    assert h.log.count(RULES_MESSAGE) == 1 + int((attempts[-1] - attempts[0]) // REGISTRATION_REMINDER_SECONDS)


def test_real_client_updated_rules_mid_run_keep_the_runtime_going():
    backend = RefreshBackend(assignments=[official_assignment("seat-1")])
    h = _real_harness(backend)
    h.tick()
    backend.valid.clear()  # the hour-long session expires...
    backend.refresh = httpx.Response(403, json={"error": "registration_incomplete",
                                                "detail": "Finish your tournament registration first.",
                                                "next_step": "rules"})  # ...after a Rules update

    h.tick()

    assert RULES_MESSAGE in h.log and not h.factory.processes[0].terminated

    backend.refresh = None  # the contestant accepted the new Rules
    h.tick(advance=REGISTRATION_RETRY_SECONDS)
    assert h.log[-1] == "Assignment discovery recovered."


def test_workers_get_the_runtimes_session_so_they_needn_t_sign_in_again():
    backend = OfficialBackend(assignments=[official_assignment("seat-1")])
    h = _real_harness(backend)

    h.tick()

    (process,) = h.factory.processes
    token = process.args[0].agent_session
    assert token == "sess-1" and len(backend.auth_bodies) == 1
    assert token not in repr(process.args[0])


def test_a_fake_official_without_a_session_hands_none():
    h = Harness([a("seat-1")])

    h.tick()

    assert h.factory.processes[0].args[0].agent_session is None


# -- worker: getting into the game ------------------------------------------------------------


class WorkerClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


def _clocked_worker(grant_result, run_game_fn=None, **input_fields):
    clock = WorkerClock()
    runs, officials = [], []

    def fake_run_game(game, context, contestant):
        runs.append(game)
        return types.SimpleNamespace(termination_reason="normal", returns={})

    def factory():
        officials.append(FakeWorkerOfficial(grant_result))
        return officials[-1]

    worker_input = TournamentWorkerInput("seat-1", "match-1", "werewolf", SPEC, EXEC, **input_fields)
    code = run_tournament_worker(worker_input, official_factory=factory, run_game_fn=run_game_fn or fake_run_game,
                                 sleep=clock.sleep, now=clock.now)
    return code, runs, officials, clock


GRANT_HICCUPS = [
    pytest.param(PlatformError("internal", status_code=500, error_code="internal_error"), id="seat-sign-in-500"),
    pytest.param(PlatformError("Could not reach", status_code=None), id="unreachable"),
    pytest.param(OfficialAgentError("slow down", status_code=429, error_code="rate_limited"), id="429"),
    pytest.param(OfficialAgentError("busy", status_code=503, error_code="agent_session_unavailable"), id="503"),
    pytest.param(AuthenticationError("Invalid or expired agent session token", status_code=401,
                                     error_code="invalid_agent_session"), id="fresh-session-401"),
]


@pytest.mark.parametrize("hiccup", GRANT_HICCUPS)
def test_worker_retries_a_temporary_grant_failure_within_seconds_and_plays(hiccup, capsys):
    code, runs, officials, clock = _clocked_worker([hiccup, hiccup, _grant()])

    assert code == EXIT_SUCCESS and len(runs) == 1
    assert officials[0].grants == ["seat-1"] * 3
    assert len(clock.sleeps) == 2 and 2.25 <= clock.sleeps[0] <= 3.75 and 4.5 <= clock.sleeps[1] <= 7.5
    assert "couldn't get into the game yet" in capsys.readouterr().out


def test_worker_that_never_gets_into_the_game_exits_not_connected_after_a_minute():
    code, runs, officials, clock = _clocked_worker(PlatformError("internal", status_code=500))

    assert code == EXIT_NOT_CONNECTED and runs == []
    assert FIRST_GRANT_RETRY_SECONDS <= clock.t <= FIRST_GRANT_RETRY_SECONDS + 20
    assert 5 <= len(officials[0].grants) <= 8
    assert max(clock.sleeps) <= 15 * 1.25


def test_worker_does_not_retry_a_definite_grant_refusal():
    code, _, officials, clock = _clocked_worker(
        OfficialAgentError("missing", status_code=404, error_code="assignment_not_found"))

    assert code == EXIT_UNEXPECTED and officials[0].grants == ["seat-1"] and clock.sleeps == []


def test_worker_turned_away_by_updated_rules_says_what_to_do(capsys):
    code, runs, _, clock = _clocked_worker(RULES_UPDATED)

    assert code == EXIT_REGISTRATION_INCOMPLETE and runs == [] and clock.sleeps == []
    assert RULES_MESSAGE in capsys.readouterr().out


def test_worker_uses_the_handed_over_session():
    code, _, officials, _ = _clocked_worker(_grant(), agent_session="sess-7")

    assert code == EXIT_SUCCESS and officials[0].session_tokens == ["sess-7"]


def test_worker_without_a_handed_over_session_signs_in_itself():
    _, _, officials, _ = _clocked_worker(_grant())

    assert officials[0].session_tokens == []


@pytest.mark.parametrize(
    "error, expected",
    [
        pytest.param(OfficialAgentError("held", status_code=409, error_code="seat_busy"), EXIT_SEAT_BUSY,
                     id="seat-busy"),
        pytest.param(OfficialAgentError("ended", status_code=409, error_code="assignment_not_grantable"), EXIT_SUCCESS,
                     id="game-ended"),
    ],
)
def test_worker_maps_a_regrant_refusal_during_play(error, expected):
    def regrant_refused(game, context, contestant):
        game.contacted = True
        raise error

    code, _, _, _ = _clocked_worker(_grant(), run_game_fn=regrant_refused)

    assert code == expected


def test_the_game_session_counts_as_reached_only_after_gameapi_answers(monkeypatch):
    import threading

    from altruagent import mcp_game
    from altruagent.worker import _LeaseGuardedGameSession

    answers = [MCPToolError("down", status_code=502), {"session_id": "game-1", "state_version": 0}]

    def fake_call_tool(client, url, tool, arguments):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(mcp_game, "call_tool", fake_call_tool)
    game = _LeaseGuardedGameSession(object(), lost=threading.Event(), session_id="game-1",
                                    game_server_url="https://gameapi.example.test")

    with pytest.raises(MCPToolError):
        game.get_state()
    assert game.contacted is False
    game.get_state()
    assert game.contacted is True


# -- a key that stopped being accepted (rotated or revoked) while running --------------------

KEY_REFUSED = OfficialAgentError(
    "The Official Agent Key was not accepted. Check that your agent is set to Self-hosted ...",
    status_code=401, error_code="invalid_official_agent_key", detail="Invalid official agent key",
)


def test_worker_refused_the_key_says_so_and_exits_with_its_own_code(capsys):
    # Was: "unexpected error: OfficialAgentError(...)" and exit 2, retried with backoff.
    code, runs, officials, clock = _clocked_worker(KEY_REFUSED)

    assert code == EXIT_KEY_REFUSED and runs == [] and clock.sleeps == []
    assert officials[0].grants == ["seat-1"]
    out = capsys.readouterr().out
    assert "The Official Agent Key was not accepted." in out and "unexpected error" not in out


def test_worker_refused_the_key_during_play_exits_with_its_own_code():
    def regrant_refused(game, context, contestant):
        game.contacted = True
        raise KEY_REFUSED

    code, _, _, _ = _clocked_worker(_grant(), run_game_fn=regrant_refused)

    assert code == EXIT_KEY_REFUSED


def test_a_failed_session_mint_is_not_a_refused_key():
    code, _, _, _ = _clocked_worker(MINT_FAILURE)

    assert code == EXIT_NOT_CONNECTED


def test_after_a_refused_key_no_new_game_is_started_and_it_is_said_once():
    h = Harness([a("seat-1"), a("seat-2"), a("seat-3")])
    h.tick()
    h.factory.processes[0].finish(EXIT_KEY_REFUSED)
    h.factory.processes[1].finish(EXIT_KEY_REFUSED)

    h.tick()
    h.official.listings = [[a("seat-1"), a("seat-2"), a("seat-3"), a("seat-4")]]
    for _ in range(10):
        h.tick(advance=120.0)  # well past every retry pause

    assert h.started() == ["seat-1", "seat-2", "seat-3"]  # seat-4 and the retries never start
    assert h.log.count(KEY_REFUSED_MESSAGE) == 1
    assert "put your new key in .env" in KEY_REFUSED_MESSAGE.lower() and "restart" in KEY_REFUSED_MESSAGE
    assert not h.factory.processes[2].terminated  # a game already running keeps playing
    h.factory.processes[2].finish(EXIT_SUCCESS)
    h.tick()
    assert h.log[-1] == "Match finished."  # not "Waiting for your next game...": none will start
    assert WAITING_MESSAGE not in h.log


def test_worker_tells_the_agent_the_game_ended_and_how_it_went(capsys):
    calls = []

    def fake_run_game(game, context, contestant):
        contestant.on_game_end = lambda final, ctx, outcome: calls.append((final.termination_reason, ctx.seat_position, outcome))
        return types.SimpleNamespace(termination_reason="completed", returns={"synthetic-1": 1.0})

    code, _, _ = _worker(run_game_fn=fake_run_game)
    assert code == EXIT_SUCCESS and calls == [("completed", 1, "won")]


def test_a_failing_on_game_end_does_not_fail_the_worker(capsys):
    def fake_run_game(game, context, contestant):
        def boom(final, ctx, outcome):
            raise RuntimeError("disk full")
        contestant.on_game_end = boom
        return types.SimpleNamespace(termination_reason="completed", returns={"synthetic-1": 1.0})

    code, _, _ = _worker(run_game_fn=fake_run_game)
    out = capsys.readouterr().out
    assert code == EXIT_SUCCESS and "on_game_end raised" in out and "finished: your agent" in out
