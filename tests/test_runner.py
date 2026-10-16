"""Unit tests for altruagent.runner (run_game / run_match) — MCP-first.

All gameplay is driven through a lightweight FakeMCPGameSession test double —
run_game only needs an object exposing get_state()/wait_for_update()/
get_legal_actions()/play_action()/send_message()/get_messages()/resign()/
get_result(), so there
is no need to mock HTTP for these; MCPGameSession's own transport is covered
by tests/test_mcp_game.py and tests/test_mcp_transport.py.
"""

from __future__ import annotations

import pytest

from altruagent.mcp_transport import MCPToolError
from altruagent.models import DecisionContext, GameState, LegalAction, Match
from altruagent.runner import (
    RESIGN,
    TERMINATE_MESSAGING,
    DecisionError,
    SendMessage,
    UnsupportedGameFlowError,
    run_game,
    run_match,
)

SESSION_ID = "session-1"


class FakeMCPGameSession:
    """Scripted stand-in for MCPGameSession. Queue return values (or
    exception instances, raised) per method — each method pops its own
    queue, so a test only needs to set up the calls it actually expects.
    """

    def __init__(self, session_id: str = SESSION_ID) -> None:
        self.session_id = session_id
        self.game_server_url = "http://fake:8000"
        self._state_queue: list = []
        self._legal_actions_queue: list = []
        self._play_action_queue: list = []
        self._send_message_queue: list = []
        self._resign_queue: list = []
        self._result_queue: list = []
        self.play_action_calls: list[dict] = []
        self.send_message_calls: list[dict] = []
        self.wait_calls: list[dict] = []
        self.get_state_calls = 0
        self.get_legal_actions_calls = 0
        self.resign_calls = 0
        # False simulates a server that predates wait_for_update.
        self.supports_wait = True

    # -- queueing helpers (chainable) ----------------------------------

    def queue_state(self, *items) -> "FakeMCPGameSession":
        self._state_queue.extend(items)
        return self

    def queue_legal_actions(self, *items) -> "FakeMCPGameSession":
        self._legal_actions_queue.extend(items)
        return self

    def queue_play_action(self, *items) -> "FakeMCPGameSession":
        self._play_action_queue.extend(items)
        return self

    def queue_send_message(self, *items) -> "FakeMCPGameSession":
        self._send_message_queue.extend(items)
        return self

    def queue_resign(self, *items) -> "FakeMCPGameSession":
        self._resign_queue.extend(items)
        return self

    def queue_result(self, *items) -> "FakeMCPGameSession":
        self._result_queue.extend(items)
        return self

    @staticmethod
    def _pop(queue: list):
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    # -- MCPGameSession-shaped interface --------------------------------

    def get_state(self) -> GameState:
        self.get_state_calls += 1
        return self._pop(self._state_queue)

    def wait_for_update(
        self,
        *,
        since_version,
        since_message_seq=None,
        since_is_current_actor=None,
        since_phase=None,
        timeout_seconds=None,
    ) -> GameState:
        """Pops from the same state queue as get_state — a wait's result is
        just the next state.
        """
        self.wait_calls.append(
            {
                "since_version": since_version,
                "since_message_seq": since_message_seq,
                "since_is_current_actor": since_is_current_actor,
                "since_phase": since_phase,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self.supports_wait:
            raise MCPToolError(
                "MCP tool 'wait_for_update' failed at the protocol level: "
                "Unknown tool: wait_for_update",
                status_code=None,
                error_code=None,
                protocol_error=True,  # as mcp_transport raises it
            )
        return self._pop(self._state_queue)

    def get_legal_actions(self) -> dict:
        self.get_legal_actions_calls += 1
        return self._pop(self._legal_actions_queue)

    def play_action(self, *, action_id=None, action=None, state_version) -> dict:
        self.play_action_calls.append(
            {"action_id": action_id, "action": action, "state_version": state_version}
        )
        return self._pop(self._play_action_queue)

    def send_message(self, *, message_type: str, content=None, recipients=None) -> dict:
        self.send_message_calls.append(
            {"message_type": message_type, "content": content, "recipients": recipients}
        )
        return self._pop(self._send_message_queue)

    def get_messages(self, *, since: int = -1) -> dict:
        raise AssertionError("get_messages not exercised by these tests")

    def resign(self) -> dict:
        self.resign_calls += 1
        return self._pop(self._resign_queue)

    def get_result(self) -> dict:
        return self._pop(self._result_queue)


def make_mcp_state(**overrides) -> GameState:
    payload = {
        "session_id": SESSION_ID,
        "game_type": "tic_tac_toe",
        "status": "in_progress",
        "state_version": 0,
        "observation": "...",
        "phase": "moving",
        "messaging_enabled": False,
        "terminated_messaging": [],
        "new_messages": [],
        "current_actor": {"agent_id": "agent-1", "position": 0},
        "is_current_actor": True,
        "is_terminal": False,
    }
    payload.update(overrides)
    return GameState.from_mcp_state(payload)


def waiting_state(**overrides) -> GameState:
    overrides.setdefault("is_current_actor", False)
    overrides.setdefault("current_actor", {"agent_id": "opponent-1", "position": 1})
    return make_mcp_state(**overrides)


def terminal_state(**overrides) -> GameState:
    overrides.setdefault("is_terminal", True)
    overrides.setdefault("is_current_actor", False)
    return make_mcp_state(**overrides)


def messaging_state(**overrides) -> GameState:
    overrides.setdefault("phase", "messaging")
    overrides.setdefault("messaging_enabled", True)
    return make_mcp_state(**overrides)


def legal_actions_result(actions: list[dict], *, state_version: int = 0) -> dict:
    return {"session_id": SESSION_ID, "state_version": state_version, "actions": actions}


def int_actions(*ints, state_version: int = 0) -> dict:
    return legal_actions_result(
        [{"action_id": str(i), "label": str(i), "input": {}} for i in ints],
        state_version=state_version,
    )


def structured_actions(*action_ids, state_version: int = 0) -> dict:
    return legal_actions_result(
        [{"action_id": aid, "label": aid, "input": {"type": "move"}} for aid in action_ids],
        state_version=state_version,
    )


def play_action_result(*, status: str = "in_progress", state_version: int = 1) -> dict:
    return {"accepted": True, "session_id": SESSION_ID, "state_version": state_version, "status": status}


def result_dict(**overrides) -> dict:
    payload = {
        "session_id": SESSION_ID,
        "is_terminal": True,
        "status": "completed",
        "returns": {"Me": 1.0, "Them": -1.0},
        "your_return": 1.0,
        "termination_reason": None,
    }
    payload.update(overrides)
    return payload


CONTEXT = DecisionContext(
    session_id=SESSION_ID, tournament_id=None, game_type="tic_tac_toe", agent_id="agent-1"
)


def no_sleep(_seconds: float) -> None:
    pass


# -- decision contract ------------------------------------------------------


def test_universal_pattern_return_first_legal_action_works():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)

    assert result.is_terminal is True
    assert game.play_action_calls == [{"action_id": "0", "action": None, "state_version": 0}]


def test_object_with_choose_action_method_works():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[-1]

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1, 2))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert result.is_terminal is True
    assert game.play_action_calls[0]["action_id"] == "2"


def test_invalid_decision_object_fails_clearly():
    game = FakeMCPGameSession()  # empty queues: get_state must never be called

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, 42, sleep=no_sleep)  # not callable, no .choose_action


def test_action_id_string_decision_accepted():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, lambda s, c: "1", sleep=no_sleep)

    assert game.play_action_calls == [{"action_id": "1", "action": None, "state_version": 0}]


def test_legal_action_object_decision_accepted():
    def choose_action(state, context):
        return next(a for a in state.legal_actions if a.action_id == "1")

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, choose_action, sleep=no_sleep)

    assert game.play_action_calls == [{"action_id": "1", "action": None, "state_version": 0}]


def test_structured_dict_decision_passed_through_verbatim():
    payload = {"type": "submit_team", "team": ["a", "b", "c", "d", "e", "f"]}
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(structured_actions("submit_team", state_version=7))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, lambda s, c: payload, sleep=no_sleep)

    assert game.play_action_calls == [{"action_id": None, "action": payload, "state_version": 7}]


def test_empty_dict_decision_rejected():
    game = FakeMCPGameSession().queue_state(make_mcp_state()).queue_legal_actions(int_actions(0, 1))

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, lambda s, c: {}, sleep=no_sleep)

    assert game.play_action_calls == []


def test_int_accepted_only_when_it_matches_a_stringified_action_id():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, lambda s, c: 1, sleep=no_sleep)

    assert game.play_action_calls == [{"action_id": "1", "action": None, "state_version": 0}]


def test_int_rejected_for_structured_game_never_guessed():
    # Pokemon-shaped action_ids ("move:0") never match a bare int — this
    # must fail clearly, not silently guess which action was meant.
    game = FakeMCPGameSession().queue_state(make_mcp_state()).queue_legal_actions(
        structured_actions("move:0", "switch:1")
    )

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, lambda s, c: 0, sleep=no_sleep)

    assert game.play_action_calls == []


def test_non_matching_string_action_id_rejected():
    game = FakeMCPGameSession().queue_state(make_mcp_state()).queue_legal_actions(int_actions(0, 1))

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, lambda s, c: "99", sleep=no_sleep)

    assert game.play_action_calls == []


def test_bool_decision_rejected_even_though_it_is_a_legal_action_value():
    # isinstance(True, int) is True in Python, and True == 1 — without an
    # explicit bool guard, returning True would silently be treated as
    # legal action "1". It must not be.
    game = FakeMCPGameSession().queue_state(make_mcp_state()).queue_legal_actions(int_actions(0, 1))

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, lambda s, c: True, sleep=no_sleep)

    assert game.play_action_calls == []


def test_unrecognized_type_decision_rejected():
    game = FakeMCPGameSession().queue_state(make_mcp_state()).queue_legal_actions(int_actions(0, 1))

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, lambda s, c: 3.14, sleep=no_sleep)

    assert game.play_action_calls == []


# -- flow ---------------------------------------------------------------


def test_make_move_invokes_decision_exactly_once_for_that_state():
    calls = []

    def choose_action(state, context):
        calls.append(state)
        return state.legal_actions[0]

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, choose_action, sleep=no_sleep)

    assert len(calls) == 1


def test_wait_for_opponent_does_not_invoke_decision_or_fetch_legal_actions():
    def choose_action(state, context):
        raise AssertionError("choose_action should not be called while waiting")

    sleep_calls = []
    game = (
        FakeMCPGameSession()
        .queue_state(waiting_state(state_version=4), terminal_state())
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, choose_action, sleep=sleep_calls.append)

    assert result.is_terminal is True
    # Long-polls the server rather than sleeping client-side.
    # Sends what it last saw, so a turn/phase change that lands before the
    # call still wakes it.
    assert game.wait_calls == [
        {
            "since_version": 4,
            "since_message_seq": None,
            "since_is_current_actor": False,
            "since_phase": "moving",
            "timeout_seconds": 20.0,
        }
    ]
    assert sleep_calls == []
    assert game.get_legal_actions_calls == 0


def test_wait_falls_back_to_sleep_when_server_lacks_wait_for_update():
    sleep_calls = []
    game = (
        FakeMCPGameSession()
        .queue_state(waiting_state(), waiting_state(), terminal_state())
        .queue_result(result_dict())
    )
    game.supports_wait = False
    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=sleep_calls.append)

    assert result.is_terminal is True
    # Tried once, then remembered the server doesn't have it.
    assert len(game.wait_calls) == 1
    assert sleep_calls == [5.0, 5.0]  # DEFAULT_WAIT_SECONDS


def test_wait_for_update_transport_error_is_retried_not_mistaken_for_an_old_server():
    game = FakeMCPGameSession().queue_state(
        waiting_state(),
        MCPToolError("transport failed", status_code=None, error_code=None),
        terminal_state(),
    ).queue_result(result_dict())

    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=no_sleep, log=lambda line: None)

    assert result.is_terminal
    assert len(game.wait_calls) == 2  # tried again, still with wait_for_update


def test_wait_passes_highest_seen_message_seq():
    state = waiting_state(
        state_version=3,
        new_messages=[{"seq": 5, "sender": 1, "content": "hi"}, {"seq": 8, "sender": 2, "content": "yo"}],
    )
    game = FakeMCPGameSession().queue_state(state, terminal_state()).queue_result(result_dict())
    run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=no_sleep)

    assert game.wait_calls[0]["since_message_seq"] == 8


def test_terminal_state_exits_cleanly_and_fetches_result():
    def choose_action(state, context):
        raise AssertionError("choose_action should not be called on a terminal state")

    game = FakeMCPGameSession().queue_state(terminal_state()).queue_result(result_dict())
    result = run_game(game, CONTEXT, choose_action, sleep=no_sleep)

    assert result.is_terminal is True
    assert result.returns == {"Me": 1.0, "Them": -1.0}
    assert game.get_legal_actions_calls == 0


def test_repeated_wait_then_make_move_transition():
    calls = []

    def choose_action(state, context):
        calls.append(state)
        return state.legal_actions[0]

    sleep_calls = []
    game = (
        FakeMCPGameSession()
        .queue_state(waiting_state(), waiting_state(), make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, choose_action, sleep=sleep_calls.append)

    assert result.is_terminal is True
    assert len(game.wait_calls) == 2
    assert sleep_calls == []
    assert len(calls) == 1


# -- resign -----------------------------------------------------------------


def test_resign_calls_resign_not_play_action():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_resign(result_dict(termination_reason="resignation"))
    )
    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=no_sleep)

    assert result.is_terminal is True
    assert result.termination_reason == "resignation"
    assert game.resign_calls == 1
    assert game.play_action_calls == []


def test_resign_still_works_when_choose_message_is_defined():
    class Agent:
        def choose_action(self, state, context):
            return RESIGN

        def choose_message(self, state, context):
            return TERMINATE_MESSAGING

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_resign(result_dict(termination_reason="resignation"))
    )
    result = run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert result.is_terminal is True
    assert game.resign_calls == 1


def test_a_werewolf_resign_waits_for_the_game_to_end():
    # Werewolf (2026-10-07): a resign takes only this agent out and the game
    # goes on (is_terminal false, eliminated true). The runner must not report
    # the game over yet: it waits like any eliminated player, never asks the
    # agent again, and returns the real final result.
    asked = []
    game = (
        FakeMCPGameSession()
        .queue_state(
            make_mcp_state(),
            make_mcp_state(eliminated=True, is_current_actor=False, state_version=1),
            terminal_state(state_version=9),
        )
        .queue_legal_actions(int_actions(0, 1))
        .queue_resign(
            result_dict(
                is_terminal=False,
                status="in_progress",
                returns=None,
                your_return=None,
                eliminated=True,
            )
        )
        .queue_result(
            result_dict(
                termination_reason="completed",
                returns={"Me": -1.0, "Them": 1.0},
                your_return=-1.0,
            )
        )
    )
    result = run_game(
        game, CONTEXT, lambda s, c: asked.append(1) or RESIGN, sleep=no_sleep
    )

    assert game.resign_calls == 1
    assert asked == [1]
    assert result.is_terminal is True
    assert result.termination_reason == "completed"
    assert result.final_result["your_return"] == -1.0
    assert len(game.wait_calls) == 1


# -- messaging ----------------------------------------------------------


def test_messaging_phase_never_invokes_choose_action():
    def choose_action(state, context):
        raise AssertionError("choose_action must not run during a messaging phase")

    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(), terminal_state())
        .queue_send_message({"accepted": True, "phase": "moving"})
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, choose_action, sleep=no_sleep)

    assert result.is_terminal is True
    assert game.play_action_calls == []


def test_default_choose_message_auto_terminates_when_absent():
    # No choose_message defined anywhere (bare function choose_action) — the
    # existing simple starter agent must still finish a messaging-enabled
    # match without any new code.
    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(), terminal_state())
        .queue_send_message({"accepted": True, "phase": "moving"})
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=no_sleep)

    assert result.is_terminal is True
    assert game.send_message_calls == [
        {"message_type": "terminate", "content": None, "recipients": None}
    ]


def test_default_choose_message_auto_terminates_for_object_without_it():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(), terminal_state())
        .queue_send_message({"accepted": True, "phase": "moving"})
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert result.is_terminal is True
    assert game.send_message_calls[0]["message_type"] == "terminate"


def test_custom_choose_message_sends_chat():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            return SendMessage("let's cooperate", recipients=[1])

    game = (
        FakeMCPGameSession()
        # The wait's result is terminal, which ends the loop after one round.
        .queue_state(messaging_state(state_version=2), terminal_state())
        .queue_send_message(
            {"accepted": True, "phase": "messaging", "message": {"seq": 11, "type": "chat"}}
        )
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert game.send_message_calls == [
        {"message_type": "chat", "content": "let's cooperate", "recipients": [1]}
    ]
    # Phase stayed "messaging" -> waits (past this agent's own message)
    # instead of busy-polling.
    assert game.wait_calls == [
        {
            "since_version": 2,
            "since_message_seq": 11,
            "since_is_current_actor": True,
            "since_phase": "messaging",
            "timeout_seconds": 20.0,
        }
    ]


def test_custom_choose_message_can_terminate_explicitly():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            return TERMINATE_MESSAGING

    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(), terminal_state())
        .queue_send_message({"accepted": True, "phase": "moving"})
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert result.is_terminal is True
    assert game.send_message_calls[0]["message_type"] == "terminate"


def test_choose_message_resolved_off_original_object_not_bound_method():
    # Regression guard: choose_message must be looked up on the object
    # create_agent() returned, not on the bound choose_action method (bound
    # methods don't proxy attribute lookups back to their owning instance).
    class Agent:
        def __init__(self):
            self.messages_sent = 0

        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            self.messages_sent += 1
            return TERMINATE_MESSAGING

    agent = Agent()
    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(), terminal_state())
        .queue_send_message({"accepted": True, "phase": "moving"})
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, agent, sleep=no_sleep)

    assert agent.messages_sent == 1


def test_repeated_messaging_rounds_across_the_match():
    # repeated_pd reopens messaging after every round — the loop must
    # handle this more than once.
    class Agent:
        def __init__(self):
            self.rounds_seen = 0

        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            self.rounds_seen += 1
            return TERMINATE_MESSAGING

    agent = Agent()
    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(), make_mcp_state(), messaging_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_send_message(
            {"accepted": True, "phase": "moving"}, {"accepted": True, "phase": "moving"}
        )
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, agent, sleep=no_sleep)

    assert result.is_terminal is True
    assert agent.rounds_seen == 2
    assert len(game.send_message_calls) == 2
    assert len(game.play_action_calls) == 1


def test_messaging_still_open_after_terminate_waits_instead_of_busy_polling():
    # This agent already terminated but others haven't — send_message's own
    # idempotent no-op still reports phase="messaging", so the runner must
    # wait rather than hammer choose_message/send_message.
    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(), terminal_state())
        .queue_send_message({"accepted": True, "phase": "messaging"})
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=no_sleep)

    assert result.is_terminal is True
    assert len(game.wait_calls) == 1
    assert len(game.send_message_calls) == 1


def test_messaging_closed_by_send_refetches_immediately_without_waiting():
    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(), terminal_state())
        .queue_send_message({"accepted": True, "phase": "moving"})
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=no_sleep)

    assert game.wait_calls == []


def test_invalid_choose_message_return_value_rejected():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            return "just chat, no wrapper"

    game = FakeMCPGameSession().queue_state(messaging_state())

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert game.send_message_calls == []


def test_choose_message_exception_chains_into_decision_error():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            raise ValueError("boom")

    game = FakeMCPGameSession().queue_state(messaging_state())

    with pytest.raises(DecisionError) as exc_info:
        run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_messaging_quota_exceeded_becomes_decision_error():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            return SendMessage("one too many")

    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state())
        .queue_send_message(MCPToolError("quota", status_code=None, error_code="MESSAGES_QUOTA_EXCEEDED"))
    )

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, Agent(), sleep=no_sleep)


def test_invalid_recipients_becomes_decision_error():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            return SendMessage("hi", recipients=[0, 1])

    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state())
        .queue_send_message(MCPToolError("bad recipients", status_code=None, error_code="INVALID_RECIPIENTS"))
    )

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, Agent(), sleep=no_sleep)


def test_messaging_runtime_unavailable_becomes_unsupported_game_flow_error():
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            return TERMINATE_MESSAGING

    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state())
        .queue_send_message(MCPToolError("unsupported", status_code=None, error_code="RUNTIME_UNAVAILABLE"))
    )

    with pytest.raises(UnsupportedGameFlowError):
        run_game(game, CONTEXT, Agent(), sleep=no_sleep)


def test_non_messaging_game_never_touches_messaging_transport():
    # Regression guard: a normal move-only game must never call send_message
    # even if choose_message is defined.
    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            raise AssertionError("choose_message should never run for this game")

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert result.is_terminal is True
    assert game.send_message_calls == []


def test_structured_game_phase_never_equals_messaging_marker():
    # Pokemon-shaped: phase is "draft"/"teambuild"/"moving", never
    # "messaging" — the generic runner treats any non-"messaging" phase the
    # same way (enumerate legal actions, invoke choose_action), with zero
    # adapter-specific code.
    def choose_action(state, context):
        raise AssertionError("choose_message-triggering path must not run")

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(phase="draft"), terminal_state())
        .queue_legal_actions(structured_actions("draft_pick:card-7"))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)

    assert result.is_terminal is True
    assert game.play_action_calls == [{"action_id": "draft_pick:card-7", "action": None, "state_version": 0}]


# -- state_version ------------------------------------------------------


def test_state_version_from_legal_actions_used_in_play_action_not_stale_get_state():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(state_version=1), terminal_state())
        .queue_legal_actions(int_actions(0, 1, state_version=9))
        .queue_play_action(play_action_result(state_version=10))
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)

    assert game.play_action_calls == [{"action_id": "0", "action": None, "state_version": 9}]


def test_stale_state_refetches_and_does_not_replay_the_decision_blindly():
    calls = []

    def choose_action(state, context):
        calls.append(state.state_version)
        return state.legal_actions[0]

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), make_mcp_state(state_version=5), terminal_state())
        .queue_legal_actions(int_actions(0, 1, state_version=0), int_actions(0, 1, state_version=5))
        .queue_play_action(
            MCPToolError("stale", status_code=None, error_code="STALE_STATE"), play_action_result()
        )
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, choose_action, sleep=no_sleep)

    assert result.is_terminal is True
    assert calls == [0, 5]  # re-decided against fresh state, not replayed
    assert len(game.play_action_calls) == 2


def test_not_your_turn_race_refetches_state_instead_of_blaming_contestant():
    calls = []

    def choose_action(state, context):
        calls.append(state)
        return state.legal_actions[0]

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(MCPToolError("stale read", status_code=None, error_code="NOT_YOUR_TURN"))
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, choose_action, sleep=no_sleep)

    assert result.is_terminal is True
    assert len(calls) == 1  # not re-invoked — this was never a contestant bug


def test_game_already_complete_race_treated_as_natural_completion():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), terminal_state(termination_reason="completed"))
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(MCPToolError("done", status_code=None, error_code="GAME_ALREADY_COMPLETE"))
        .queue_result(result_dict(termination_reason="completed"))
    )
    result = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)

    assert result.is_terminal is True
    assert result.termination_reason == "completed"


def test_the_final_state_keeps_the_servers_whole_result():
    # get_result also says how this agent did (your_return, the game's own
    # per-player result, the winner): the worker reports won/lost from it.
    answer = result_dict(termination_reason="completed", winner_agent_id="agent-1",
                         result={"seats": {"agent-1": {"outcome": "win"}}})
    game = FakeMCPGameSession().queue_state(terminal_state()).queue_result(answer)

    final = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)

    assert final.final_result == answer
    assert make_mcp_state().final_result is None


def test_action_runtime_unavailable_becomes_unsupported_game_flow_error():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(MCPToolError("unsupported", status_code=None, error_code="RUNTIME_UNAVAILABLE"))
    )

    with pytest.raises(UnsupportedGameFlowError):
        run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)


def test_invalid_action_error_becomes_decision_error():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(MCPToolError("bad action", status_code=None, error_code="INVALID_ACTION"))
    )

    with pytest.raises(DecisionError):
        run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)


def test_unknown_error_code_propagates_as_mcp_tool_error():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(MCPToolError("weird", status_code=None, error_code="SOME_OTHER_ERROR"))
    )

    with pytest.raises(MCPToolError) as exc_info:
        run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)

    assert exc_info.value.error_code == "SOME_OTHER_ERROR"


# -- errors -----------------------------------------------------------------


def test_contestant_exception_chains_into_decision_error():
    def choose_action(state, context):
        raise ValueError("boom")

    game = FakeMCPGameSession().queue_state(make_mcp_state()).queue_legal_actions(int_actions(0, 1))

    with pytest.raises(DecisionError) as exc_info:
        run_game(game, CONTEXT, choose_action, sleep=no_sleep)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "boom" in str(exc_info.value.__cause__)


# -- context ------------------------------------------------------------


def test_context_is_passed_through_to_decision_fn():
    received = []

    def choose_action(state, context):
        received.append(context)
        return state.legal_actions[0]

    context = DecisionContext(
        session_id="session-42", tournament_id="tournament-7", game_type="tic_tac_toe", agent_id="agent-99"
    )
    game = (
        FakeMCPGameSession(session_id="session-42")
        .queue_state(make_mcp_state(session_id="session-42"), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    run_game(game, context, choose_action, sleep=no_sleep)

    assert len(received) == 1
    assert received[0].session_id == "session-42"
    assert received[0].tournament_id == "tournament-7"
    assert received[0].game_type == "tic_tac_toe"
    assert received[0].agent_id == "agent-99"


def test_run_match_builds_context_from_match_and_delegates_to_game():
    match = Match.from_dict(
        {
            "session_id": "session-77",
            "status": "in_progress",
            "tournament_id": "tournament-3",
            "game_type": "tic_tac_toe",
        }
    )
    fake_game = (
        FakeMCPGameSession(session_id="session-77")
        .queue_state(make_mcp_state(session_id="session-77"), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    match.game = lambda: fake_game  # bypass real resolution; tested in test_sessions.py

    received = []

    def choose_action(state, context):
        received.append(context)
        return state.legal_actions[0]

    result = run_match(match, "agent-1", choose_action, sleep=no_sleep)

    assert result.is_terminal is True
    assert received[0].session_id == "session-77"
    assert received[0].tournament_id == "tournament-3"
    assert received[0].game_type == "tic_tac_toe"
    assert received[0].agent_id == "agent-1"


def test_run_match_never_constructs_a_rest_gamesession():
    # Proves no hidden REST fallback at the run_match level: match.game()
    # (not match.rest_game()) is the only thing run_match calls.
    match = Match.from_dict({"session_id": "s-1", "status": "in_progress", "game_type": "tic_tac_toe"})
    fake_game = (
        FakeMCPGameSession(session_id="s-1")
        .queue_state(make_mcp_state(session_id="s-1"), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    game_calls = []
    match.game = lambda: (game_calls.append(1) or fake_game)

    def rest_game_should_not_be_called():
        raise AssertionError("run_match must never call Match.rest_game()")

    match.rest_game = rest_game_should_not_be_called

    run_match(match, "agent-1", lambda s, c: s.legal_actions[0], sleep=no_sleep)

    assert game_calls == [1]


# -- current MCP payloads: embedded legal_actions, post-move state ------------


def embedded(state_version: int, *ints) -> dict:
    return int_actions(*ints, state_version=state_version)


def test_embedded_legal_actions_used_without_get_legal_actions_call():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(state_version=3, legal_actions=embedded(3, 0, 1)), terminal_state())
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, lambda s, c: s.legal_actions[1], sleep=no_sleep)

    assert game.get_legal_actions_calls == 0
    assert game.play_action_calls == [{"action_id": "1", "action": None, "state_version": 3}]


def test_play_action_post_move_state_used_without_refetch():
    post_move = {
        "session_id": SESSION_ID,
        "state_version": 4,
        "phase": "moving",
        "is_current_actor": True,
        "is_terminal": False,
        "legal_actions": embedded(4, 7),
    }
    result = dict(play_action_result(state_version=4), state=post_move)
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(state_version=3, legal_actions=embedded(3, 0)), terminal_state())
        .queue_play_action(result, play_action_result(status="completed"))
        .queue_result(result_dict())
    )
    run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)

    # Second move came straight from the first play_action's `state`.
    assert game.play_action_calls == [
        {"action_id": "0", "action": None, "state_version": 3},
        {"action_id": "7", "action": None, "state_version": 4},
    ]
    # Initial read + one after the final (completed, state-less) move.
    assert game.get_state_calls == 2


def test_eliminated_agent_never_acts_or_chats_and_waits_for_the_end():
    def never(state, context):
        raise AssertionError("an eliminated agent must not be asked to decide")

    class Agent:
        choose_action = staticmethod(never)
        choose_message = staticmethod(never)

    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(eliminated=True, is_current_actor=False), terminal_state())
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, Agent(), sleep=no_sleep)

    assert result.is_terminal is True
    assert game.send_message_calls == []
    assert len(game.wait_calls) == 1


def test_player_eliminated_race_refetches_instead_of_crashing():
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(legal_actions=embedded(0, 0)), terminal_state())
        .queue_play_action(
            MCPToolError("eliminated", status_code=None, error_code="PLAYER_ELIMINATED")
        )
        .queue_result(result_dict())
    )
    result = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=no_sleep)

    assert result.is_terminal is True


# -- temporary failures are retried in place (no lost turns, no restart) -----------------

from altruagent import runner as runner_module  # noqa: E402
from altruagent.errors import AuthenticationError, PlatformError  # noqa: E402
from altruagent.official import OfficialAgentError  # noqa: E402


class Clock:
    """A fake monotonic clock that a fake sleep advances."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def _transport_error(text="ConnectError('connection reset')"):
    return MCPToolError(f"MCP tool 'get_game_state' failed: {text}", status_code=None, error_code=None)


TRANSIENT_ERRORS = [
    pytest.param(_transport_error(), id="dropped-connection"),
    pytest.param(MCPToolError("MCP tool request failed with HTTP 502", status_code=502, error_code=None), id="http-502"),
    pytest.param(MCPToolError("MCP tool request failed with HTTP 504", status_code=504, error_code=None), id="http-504"),
    pytest.param(MCPToolError("MCP tool request failed with HTTP 429", status_code=429, error_code=None), id="http-429"),
    pytest.param(MCPToolError("busy", status_code=None, error_code="RUNTIME_TEMPORARILY_UNAVAILABLE"), id="runtime-busy"),
    pytest.param(MCPToolError("timed out", status_code=None, error_code=None), id="read-timeout"),
    # A GameAPI 401 whose seat re-grant hit a control-plane hiccup.
    pytest.param(PlatformError("Request failed with status 503.", status_code=503), id="regrant-503"),
    pytest.param(PlatformError("Could not reach /grant: ConnectError", status_code=None), id="regrant-unreachable"),
    pytest.param(OfficialAgentError("busy", status_code=503, error_code="agent_session_unavailable"), id="regrant-session"),
    pytest.param(OfficialAgentError("slow down", status_code=429, error_code="rate_limited"), id="regrant-429"),
]


@pytest.mark.parametrize("error", TRANSIENT_ERRORS)
def test_a_temporary_failure_while_waiting_is_retried_and_the_game_goes_on(error):
    clock = Clock()
    log: list[str] = []
    game = (
        FakeMCPGameSession()
        .queue_state(waiting_state(), error, make_mcp_state(state_version=2), waiting_state(state_version=3),
                     terminal_state())
        .queue_legal_actions(int_actions(0, 1, state_version=2))
        .queue_play_action(play_action_result(state_version=3))
        .queue_result(result_dict())
    )

    result = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=clock.sleep, now=clock.now,
                      log=log.append)

    assert result.is_terminal
    assert len(game.play_action_calls) == 1
    assert len(clock.sleeps) == 1 and 0.5 <= clock.sleeps[0] <= 1.0
    assert log[0].startswith("Connection problem (") and log[-1] == "Connection back; the game goes on."
    assert len(log) == 2


@pytest.mark.parametrize("error", TRANSIENT_ERRORS[:3])
def test_a_temporary_failure_on_the_first_read_is_retried(error):
    clock = Clock()
    game = FakeMCPGameSession().queue_state(error, error, terminal_state()).queue_result(result_dict())

    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert result.is_terminal and game.get_state_calls == 3
    assert len(clock.sleeps) == 2 and 1.0 <= clock.sleeps[1] <= 2.0  # the pause grows


def test_a_move_that_failed_in_transit_is_never_resent_blindly():
    # The answer was lost; the move may or may not have landed. The runner
    # re-reads the state and asks the agent again from what it sees.
    clock = Clock()
    decisions = []
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(state_version=0), make_mcp_state(state_version=0), terminal_state())
        .queue_legal_actions(int_actions(0, 1), int_actions(0, 1))
        .queue_play_action(_transport_error(), play_action_result(status="completed"))
        .queue_result(result_dict())
    )

    def choose(state, context):
        decisions.append(state.state_version)
        return state.legal_actions[0]

    run_game(game, CONTEXT, choose, sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert decisions == [0, 0]  # decided again from a fresh read, not replayed
    assert game.get_state_calls == 3 and len(game.play_action_calls) == 2


def test_a_move_that_landed_before_its_answer_was_lost_is_not_played_twice():
    clock = Clock()
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(state_version=0), waiting_state(state_version=1), terminal_state())
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(MCPToolError("HTTP 504", status_code=504, error_code=None))
        .queue_result(result_dict())
    )

    run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert len(game.play_action_calls) == 1


def test_runtime_temporarily_unavailable_on_a_turn_based_move_is_retried_and_reported():
    clock = Clock()
    results = []

    class Agent:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def on_action_result(self, result, context):
            results.append(result)

    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(), make_mcp_state(), terminal_state())
        .queue_legal_actions(int_actions(0), int_actions(0))
        .queue_play_action(MCPToolError("engine busy", status_code=None, error_code="RUNTIME_TEMPORARILY_UNAVAILABLE"),
                           play_action_result(status="completed"))
        .queue_result(result_dict())
    )

    run_game(game, CONTEXT, Agent(), sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert results[0] == {"error": "RUNTIME_TEMPORARILY_UNAVAILABLE", "detail": "engine busy"}
    assert len(game.play_action_calls) == 2


def test_a_chat_message_that_failed_in_transit_is_not_resent_blindly():
    clock = Clock()
    sent = []

    class Talker:
        def choose_action(self, state, context):
            return state.legal_actions[0]

        def choose_message(self, state, context):
            sent.append(state.state_version)
            return SendMessage("hello")

    game = (
        FakeMCPGameSession()
        .queue_state(messaging_state(state_version=4), waiting_state(state_version=5), terminal_state())
        .queue_send_message(_transport_error())
        .queue_result(result_dict())
    )

    run_game(game, CONTEXT, Talker(), sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert sent == [4] and len(game.send_message_calls) == 1


def test_the_terminal_result_read_is_retried_too():
    clock = Clock()
    game = FakeMCPGameSession().queue_state(terminal_state()).queue_result(_transport_error(), result_dict())

    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert result.returns == {"Me": 1.0, "Them": -1.0}


def test_failures_that_last_too_long_reach_the_worker():
    clock = Clock()
    error = _transport_error()
    game = FakeMCPGameSession().queue_state(waiting_state(), *([error] * 200))

    with pytest.raises(MCPToolError) as exc_info:
        run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert exc_info.value is error
    assert runner_module.TRANSIENT_GIVE_UP_SECONDS <= clock.t < runner_module.TRANSIENT_GIVE_UP_SECONDS + 10
    assert max(clock.sleeps) <= runner_module.TRANSIENT_RETRY_MAX_SECONDS
    assert len(clock.sleeps) > 15  # tried every few seconds, not once a minute


def test_the_give_up_clock_starts_over_after_any_successful_call():
    clock = Clock()
    error = _transport_error()
    # 80 s of failures, one good read, then 80 s more: never 90 s in a row.
    burst = [error] * 20
    game = FakeMCPGameSession().queue_state(waiting_state(), *burst, waiting_state(), *burst, terminal_state())
    game.queue_result(result_dict())

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.t += 4.0

    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=sleep, now=clock.now, log=lambda line: None)

    assert result.is_terminal and clock.t >= 160


_ONE_ACTION = int_actions(0)  # embedded in the state, as GameAPI sends it


def test_a_move_that_keeps_failing_while_reads_work_still_reaches_the_worker():
    # Every read works, every move fails (say a gateway error on that one
    # tool). The re-read after each failure mustn't count as "the trouble is
    # over", or the agent would be asked and the move retried every second
    # for the rest of the game. The pauses grow and the worker gets the error
    # after TRANSIENT_GIVE_UP_SECONDS, like any other lasting failure.
    clock = Clock()
    lines: list[str] = []
    error = MCPToolError("MCP tool 'play_action' request failed with HTTP 502", status_code=502, error_code=None)
    game = (
        FakeMCPGameSession()
        .queue_state(*[make_mcp_state(legal_actions=_ONE_ACTION) for _ in range(200)])
        .queue_play_action(*([error] * 200))
    )

    with pytest.raises(MCPToolError) as exc_info:
        run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=clock.sleep, now=clock.now, log=lines.append)

    assert exc_info.value is error
    assert runner_module.TRANSIENT_GIVE_UP_SECONDS <= clock.t < runner_module.TRANSIENT_GIVE_UP_SECONDS + 10
    assert len(game.play_action_calls) < 40  # paced: about 1, 2, 4, then 5 s apart
    assert len(lines) == 1 and lines[0].startswith("Connection problem")


def test_after_a_failed_move_a_wait_that_works_ends_the_trouble():
    # The move landed but its answer was lost; the game then went on (a wait
    # worked) for longer than the give-up time. A later, unrelated blip is a
    # new spell of trouble, retried as usual, not "90 s of failure".
    clock = Clock()
    lines: list[str] = []

    class SlowOpponentGame(FakeMCPGameSession):
        def wait_for_update(self, **kwargs) -> GameState:
            clock.t += 120.0  # the opponent thinks for two minutes
            return super().wait_for_update(**kwargs)

    error = MCPToolError("MCP tool 'play_action' request failed with HTTP 504", status_code=504, error_code=None)
    game = (
        SlowOpponentGame()
        .queue_state(
            make_mcp_state(state_version=0, legal_actions=_ONE_ACTION),
            waiting_state(state_version=1),  # re-read after the first failure: it landed
            make_mcp_state(state_version=2, legal_actions=int_actions(0, state_version=2)),  # the wait's answer
            waiting_state(state_version=3),  # re-read after the second failure: it landed
            terminal_state(state_version=4),  # the wait's answer
        )
        .queue_play_action(error, error)
        .queue_result(result_dict())
    )

    result = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=clock.sleep, now=clock.now, log=lines.append)

    assert result.is_terminal and len(game.play_action_calls) == 2
    assert [line.split(" (")[0].rstrip(".") for line in lines] == [
        "Connection problem", "Connection back; the game goes on",
        "Connection problem", "Connection back; the game goes on",
    ]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(MCPToolError("gone", status_code=None, error_code="SESSION_NOT_FOUND"), id="session-not-found"),
        pytest.param(MCPToolError("bad request", status_code=400, error_code=None), id="http-400"),
        pytest.param(MCPToolError("forbidden", status_code=403, error_code=None), id="http-403"),
        pytest.param(MCPToolError("still 401 after a fresh grant", status_code=401, error_code=None), id="http-401"),
        pytest.param(OfficialAgentError("held", status_code=409, error_code="seat_busy"), id="seat-busy"),
        pytest.param(OfficialAgentError("ended", status_code=409, error_code="assignment_not_grantable"), id="ended"),
        pytest.param(AuthenticationError("odd", status_code=403), id="auth-403"),
    ],
)
def test_definite_errors_are_not_retried(error):
    clock = Clock()
    game = FakeMCPGameSession().queue_state(waiting_state(), error)

    with pytest.raises(type(error)):
        run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert clock.sleeps == []


def test_an_old_server_without_wait_for_update_still_falls_back_to_sleeping():
    clock = Clock()
    game = FakeMCPGameSession().queue_state(waiting_state(), terminal_state()).queue_result(result_dict())
    game.supports_wait = False

    run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert clock.sleeps == [5.0]  # DEFAULT_WAIT_SECONDS, no retry pauses


def _tool_error(tool="play_action"):
    # What mcp_transport raises for an isError answer: FastMCP rejected the
    # arguments, or the tool crashed. GameAPI sends game errors as results.
    return MCPToolError(
        f"MCP tool {tool!r} failed at the protocol level: Error executing tool {tool}: 1 validation error",
        status_code=None, error_code=None, protocol_error=True,
    )


def test_a_move_the_server_cant_handle_is_tried_once_more_then_reaches_the_worker():
    # Was: retried as a "Connection problem" every few seconds for 90 s, so
    # the agent was asked (an LLM agent: one model call) about 19 times, and
    # the worker was then restarted to do the same again.
    clock = Clock()
    lines: list[str] = []
    asked = []
    error = _tool_error()
    game = (
        FakeMCPGameSession()
        .queue_state(*[make_mcp_state(legal_actions=_ONE_ACTION) for _ in range(50)])
        .queue_play_action(*([error] * 50))
    )

    def choose(state, context):
        asked.append(state.state_version)
        return state.legal_actions[0]

    with pytest.raises(MCPToolError) as exc_info:
        run_game(game, CONTEXT, choose, sleep=clock.sleep, now=clock.now, log=lines.append)

    assert exc_info.value is error
    assert len(asked) == 2 and len(game.play_action_calls) == 2
    assert clock.t < 5
    assert len(lines) == 1 and lines[0].startswith("The game server couldn't handle that call (")
    assert lines[0].endswith("); trying once more.")
    assert not any("Connection problem" in line for line in lines)


def test_a_one_off_tool_error_on_a_move_is_tried_once_more_and_the_game_goes_on():
    clock = Clock()
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(legal_actions=_ONE_ACTION), make_mcp_state(legal_actions=_ONE_ACTION),
                     terminal_state())
        .queue_play_action(_tool_error(), play_action_result(status="completed"))
        .queue_result(result_dict())
    )

    result = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=clock.sleep, now=clock.now,
                      log=lambda line: None)

    assert result.is_terminal and len(game.play_action_calls) == 2


@pytest.mark.parametrize("tool", ["wait_for_update", "get_game_state"])
def test_a_read_the_server_cant_answer_for_a_while_is_retried_and_the_game_goes_on(tool):
    # A read's arguments are built by the runner, and GameAPI answers every
    # game error as a normal result, so an isError answer to a read is the
    # server's own trouble (a crash while its engine restarts, say). Was:
    # tried once more, then the worker stopped and sat out at least 60 s.
    clock = Clock()
    lines: list[str] = []
    error = _tool_error(tool)
    game = FakeMCPGameSession()
    if tool == "get_game_state":
        game.queue_state(error, error, error, waiting_state(), terminal_state())
    else:
        game.queue_state(waiting_state(), error, error, error, terminal_state())
    game.queue_result(result_dict())

    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=clock.sleep, now=clock.now, log=lines.append)

    assert result.is_terminal and len(clock.sleeps) == 3
    assert lines[0].startswith("The game server couldn't answer (") and lines[0].endswith("; retrying for up to 90s.")
    assert lines[-1] == "The game server answers again; the game goes on."
    assert len(lines) == 2 and not any("Connection problem" in line for line in lines)


def test_a_read_the_server_never_answers_reaches_the_worker_after_the_give_up_time():
    clock = Clock()
    asked = []
    error = _tool_error("wait_for_update")
    game = FakeMCPGameSession().queue_state(waiting_state(), *([error] * 200))

    with pytest.raises(MCPToolError) as exc_info:
        run_game(game, CONTEXT, lambda s, c: asked.append(1) or RESIGN, sleep=clock.sleep, now=clock.now,
                 log=lambda line: None)

    assert exc_info.value is error and exc_info.value.protocol_error
    assert runner_module.TRANSIENT_GIVE_UP_SECONDS <= clock.t < runner_module.TRANSIENT_GIVE_UP_SECONDS + 10
    assert len(game.wait_calls) < 40  # paced: about 1, 2, 4, then 5 s apart
    assert asked == []  # reads never ask the agent


def test_after_a_move_the_server_cant_handle_a_read_it_cant_answer_is_retried():
    # The move is tried once more only; the re-read that follows is a read.
    clock = Clock()
    move_error, read_error = _tool_error(), _tool_error("get_game_state")
    game = (
        FakeMCPGameSession()
        .queue_state(make_mcp_state(legal_actions=_ONE_ACTION), read_error, read_error,
                     make_mcp_state(legal_actions=_ONE_ACTION), terminal_state())
        .queue_play_action(move_error, play_action_result(status="completed"))
        .queue_result(result_dict())
    )

    result = run_game(game, CONTEXT, lambda s, c: s.legal_actions[0], sleep=clock.sleep, now=clock.now,
                      log=lambda line: None)

    assert result.is_terminal and len(game.play_action_calls) == 2


def test_each_success_allows_one_more_try_after_a_tool_error():
    # One-off tool errors on separate turns, each followed by a call that
    # works: never two in a row, so the game goes on.
    clock = Clock()
    error = _tool_error("wait_for_update")
    game = (
        FakeMCPGameSession()
        .queue_state(waiting_state(), error, waiting_state(state_version=1), error, terminal_state())
        .queue_result(result_dict())
    )

    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert result.is_terminal and len(game.wait_calls) == 4


def test_too_many_waits_from_an_abandoned_wait_is_paced_and_retried():
    # GameAPI caps the wait_for_update calls waiting at once per agent and
    # game. One this runtime gave up on (a dropped connection) can still be
    # waiting there for up to 25 s, so the next ones may be refused for a
    # moment. Not the agent's fault: pause, then wait again.
    clock = Clock()
    busy = MCPToolError("You already have 2 wait_for_update calls waiting in this game.",
                        status_code=None, error_code="TOO_MANY_WAITS")
    game = (
        FakeMCPGameSession()
        .queue_state(waiting_state(), busy, busy, terminal_state())
        .queue_result(result_dict())
    )

    result = run_game(game, CONTEXT, lambda s, c: RESIGN, sleep=clock.sleep, now=clock.now, log=lambda line: None)

    assert result.is_terminal
    assert len(game.wait_calls) == 3 and len(clock.sleeps) == 2


# -- a move that can't be sent as JSON is the agent's own bug, not a connection problem --


class _NumpyLikeInt:
    """Stands in for numpy.int64 or a contestant's own class in a move."""


def _real_transport_game(handler):
    """A real MCPGameSession whose calls go through the real
    mcp_transport.call_tool and MCP SDK, over a mocked HTTP layer."""
    from functools import partial

    from altruagent import mcp_game as mcp_game_module
    from altruagent.mcp_game import MCPGameSession
    from test_mcp_transport import FakeClient, make_factory

    game = MCPGameSession(FakeClient(), session_id=SESSION_ID, game_server_url="http://game.example.test")
    real_call_tool = partial(mcp_game_module.call_tool, httpx_client_factory=make_factory(handler))
    game._call = lambda tool, arguments: real_call_tool(game._client, game._mcp_url(), tool, arguments)
    return game


def test_a_move_that_cant_be_sent_as_json_fails_at_once_as_the_agents_error():
    # Was: classified as "no answer at all", so for 90 s the runner logged
    # "Connection problem", asked the agent again every 1-5 s (about 26
    # decisions, each an LLM call), then the worker said it had lost the
    # connection. The same bug failed at once before this package.
    import json as json_module

    import httpx

    from test_mcp_transport import _healthy_handler, _tool_call_response

    clock = Clock()
    lines: list[str] = []
    asked = []
    tools_called = []
    state = {
        "session_id": SESSION_ID, "game_type": "redalert", "status": "in_progress", "state_version": 4,
        "phase": "moving", "is_current_actor": True, "is_terminal": False,
        "legal_actions": {"session_id": SESSION_ID, "state_version": 4, "actions": []},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json_module.loads(request.content)
        if body.get("method") == "tools/call":
            tools_called.append(body["params"]["name"])
            return httpx.Response(200, json=_tool_call_response(body.get("id"), state))
        return _healthy_handler(request)

    def choose(state, context):
        asked.append(state.state_version)
        return {"type": "move", "unit": _NumpyLikeInt()}

    with pytest.raises(DecisionError) as exc_info:
        run_game(_real_transport_game(handler), CONTEXT, choose, sleep=clock.sleep, now=clock.now, log=lines.append)

    assert "can't be sent" in str(exc_info.value) and "plain Python values" in str(exc_info.value)
    assert asked == [4] and clock.sleeps == []
    assert tools_called == ["get_game_state"]  # the move never left the process
    assert lines == []  # no "Connection problem"


def test_a_message_that_cant_be_sent_as_json_fails_at_once_as_the_agents_error():
    clock = Clock()
    lines: list[str] = []
    unsendable = MCPToolError("MCP tool 'send_message' can't send its arguments as JSON: Unable to serialize",
                              status_code=None, error_code=None, local_error=True)

    class Talker:
        def choose_action(self, state, context):
            raise AssertionError("not reached")

        def choose_message(self, state, context):
            return SendMessage("hello", recipients=[1])

    game = FakeMCPGameSession().queue_state(messaging_state()).queue_send_message(unsendable)

    with pytest.raises(DecisionError) as exc_info:
        run_game(game, CONTEXT, Talker(), sleep=clock.sleep, now=clock.now, log=lines.append)

    assert exc_info.value.__cause__ is unsendable and "can't be sent" in str(exc_info.value)
    assert len(game.send_message_calls) == 1 and clock.sleeps == [] and lines == []


def _chat(seq, sender, text, recipients=()):
    return {"seq": seq, "sender": sender, "recipients": list(recipients), "content": text, "type": "chat"}


class _ChatThenVote:
    def __init__(self):
        self.seen = []

    def choose_message(self, state, context):
        return TERMINATE_MESSAGING

    def choose_action(self, state, context):
        self.seen.append([m.content for m in state.new_messages])
        return state.legal_actions[0]


def test_messages_said_after_our_last_read_reach_the_vote():
    # Werewolf: new_messages restarts when the phase flips, so the end of a discussion used to
    # vanish before the vote. The runner now fetches it with get_messages(since=...).
    class Game(FakeMCPGameSession):
        def __init__(self):
            super().__init__()
            self.since = []

        def get_messages(self, *, since=-1):
            self.since.append(since)
            closing = [_chat(0, 1, "old"), _chat(1, 2, "I'm the seer!"), _chat(2, 3, "psst", [4])]
            return {"messages": [m for m in closing if m["seq"] > since]}

    game = (
        Game()
        .queue_state(messaging_state(your_position=0, new_messages=[_chat(0, 1, "old")]),
                     make_mcp_state(your_position=0), terminal_state(your_position=0))
        .queue_send_message({"phase": "moving"})
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    agent = _ChatThenVote()
    final = run_game(game, CONTEXT, agent, sleep=no_sleep)
    assert agent.seen == [["I'm the seer!"]]  # a private message between two others is never shown
    assert game.since[0] == 0 and [m.content for m in final.new_messages] == []  # nothing new at the end


def test_a_failing_catch_up_never_costs_the_game():
    class Game(FakeMCPGameSession):
        def get_messages(self, *, since=-1):
            raise RuntimeError("boom")

    game = (
        Game()
        .queue_state(messaging_state(new_messages=[_chat(0, 1, "hi")]), make_mcp_state(), terminal_state())
        .queue_send_message({"phase": "moving"})
        .queue_legal_actions(int_actions(0, 1))
        .queue_play_action(play_action_result())
        .queue_result(result_dict())
    )
    assert run_game(game, CONTEXT, _ChatThenVote(), sleep=no_sleep).is_terminal
