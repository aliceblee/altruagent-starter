"""Tests for the general-purpose example LLM agent (examples/llm_agent.py),
its Pokémon structured-action adapters (examples/llm/pokemon.py), and its
provider layer (examples/llm/providers.py).

The model is always mocked — a scripted ``FakeProvider`` or an
``httpx.MockTransport`` behind the real ``OpenAIProvider`` — so no test ever
reaches OpenAI or uses a real key. Game payloads mirror Agent_ACP GameAPI's
shapes: Werewolf from ``runtime_adapters/openspiel_adapter.py`` (plain
``{action_id, label}`` legal actions, ``players``/``your_position``/
``game_state``/``new_messages`` state), Pokémon from ``pokemon_adapter``
(see tests/test_smoke_agent.py).
"""

from __future__ import annotations

import json

import httpx
import pytest

from altruagent import RESIGN, TERMINATE_MESSAGING, DecisionContext, SendMessage, WithReasoning
from altruagent.mcp_game import MCPGameSession
from altruagent.models import GameState, LegalAction
from altruagent.runner import DecisionError, _validate_decision, run_game
from examples import llm_agent, smoke_agent
from examples.llm import providers
from examples.llm.base import Choice, InvalidChoice, UnsupportedStructuredAction, object_schema
from examples.llm.providers import OpenAIProvider, ProviderError, provider_from_env
from examples.llm_agent import LLMAgent, create_agent

POKEMON = DecisionContext(session_id="s", tournament_id=None, game_type="pokemon_vgc_doubles_draft",
                          agent_id="a", seat_position=0)
WEREWOLF = DecisionContext(session_id="w", tournament_id=None, game_type="werewolf", agent_id="a2", seat_position=2)
ROSTER = ["Incineroar", "Rillaboom", "Urshifu", "Amoonguss", "Tornadus", "Flutter Mane"]
PASS = {"type": "pass"}
FAKE_KEY = "sk-test-DO-NOT-LEAK-1234567890"


# -- fixtures ------------------------------------------------------------------------------


class FakeProvider:
    """Returns scripted answers in order (a dict, or an exception to raise)."""

    model = "fake-model"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []

    def complete_structured(self, messages, schema_name, schema):
        self.calls.append({"messages": [dict(m) for m in messages], "schema_name": schema_name, "schema": schema})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def prompt(self, index=0) -> dict:
        return json.loads(self.calls[index]["messages"][1]["content"])


def _agent(provider, log=None, **kwargs):
    lines = log if log is not None else []
    return LLMAgent(provider, log=lines.append, **kwargs)


def _answer(**fields):
    fields.setdefault("reasoning_summary", "Because it is good.")
    return fields


def _raw_state(*legal, phase="moving", my_turn=True, version=3, terminal=False, **extra):
    state = {
        "session_id": "s",
        "status": "completed" if terminal else "in_progress",
        "phase": phase,
        "is_current_actor": my_turn,
        "is_terminal": terminal,
        "state_version": version,
        "observation": {"turn": 1},
        **extra,
    }
    if my_turn and legal:
        state["legal_actions"] = {"session_id": "s", "state_version": version, "actions": list(legal)}
    return state


def _state(*legal, **kwargs) -> GameState:
    return GameState.from_mcp_state(_raw_state(*legal, **kwargs))


# Pokémon payloads


def _templated(action_id, template):
    return {"action_id": action_id, "label": action_id, "input": {"action_id": action_id, "action": template}}


def _draft(*card_ids):
    return [
        {"action_id": f"draft_pick:{c}", "label": f"Draft {c}",
         "input": {"action_id": f"draft_pick:{c}", "action": {"type": "draft_pick", "card_id": c}}}
        for c in card_ids
    ]


def _lineup():
    return _templated("select_lineup", {"type": "select_lineup", "roster": ROSTER, "bring_count": 4,
                                         "lead_count": 2, "instructions": "Choose exactly 4..."})


def _move(move_id, targets):
    return {"type": "move", "move_id": move_id, "base_power": 80, "category": "physical",
            "move_type": "normal", "current_pp": 10, "accuracy": 100, "targets": targets}


def _switch(species):
    return {"type": "switch", "species": species}


def _doubles(slot_0, slot_1, forced=(False, False)):
    slots = [
        {"slot": 0, "board_position": -1, "active": {"species": "Incineroar"}, "force_switch": forced[0], "options": slot_0},
        {"slot": 1, "board_position": -2, "active": {"species": "Rillaboom"}, "force_switch": forced[1], "options": slot_1},
    ]
    return _templated("doubles_turn", {"type": "doubles_turn", "slots": slots, "target_legend": {"1": "opp A"},
                                        "instructions": "Submit one action per slot..."})


# Werewolf payloads

PLAYERS = [{"position": i, "name": f"Player{i}", "agent_id": f"a{i}"} for i in range(7)]


def _ww_actions(*pairs):
    return [{"action_id": aid, "label": label, "input": {"session_id": "w", "action_id": aid, "state_version": 4}}
            for aid, label in pairs]


def _ww_raw(*legal, phase="moving", sub_phase="day_vote", my_turn=True, version=4, messages=(), terminal=False):
    return _raw_state(
        *legal, phase=phase, my_turn=my_turn, version=version, terminal=terminal,
        observation="You are Player2. Your role: villager.",
        game_type="werewolf", your_position=2, players=PLAYERS, eliminated=False,
        game_state={"phase": sub_phase, "day": 1, "alive": [p["name"] for p in PLAYERS], "dead": []},
        new_messages=[{"seq": m[0], "sender": m[1], "recipients": list(m[3]) if len(m) > 3 else [], "content": m[2],
                       "type": "chat"} for m in messages],
    )


def _ww(*legal, **kwargs) -> GameState:
    return GameState.from_mcp_state(_ww_raw(*legal, **kwargs))


class ScriptedGame:
    """A fake MCPGameSession for the real run_game loop."""

    def __init__(self, states, result, send_results=()):
        self.states = list(states)
        self.result = result
        self.send_results = list(send_results)
        self.play_calls: list[dict] = []
        self.send_calls: list[dict] = []
        self.waits = 0

    def get_state(self):
        return GameState.from_mcp_state(self.states.pop(0))

    def wait_for_update(self, **kwargs):
        self.waits += 1
        return GameState.from_mcp_state(self.states.pop(0))

    def play_action(self, **kwargs):
        self.play_calls.append(kwargs)
        return {"accepted": True, "state": self.states.pop(0)}

    def send_message(self, **kwargs):
        self.send_calls.append(kwargs)
        return self.send_results.pop(0)

    def get_result(self):
        return self.result


RESULT = {"is_terminal": True, "status": "completed", "returns": {"a": 1.0}, "termination_reason": "normal"}


# -- generic core ---------------------------------------------------------------------------


def test_discrete_action_is_one_exact_legal_action_with_reasoning():
    provider = FakeProvider(_answer(action_id="east", reasoning_summary="East reaches the goal fastest."))
    legal = [{"action_id": "north", "label": "Move north"}, {"action_id": "east", "label": "Move east"}]

    decision = _agent(provider).choose_action(_state(*legal), DecisionContext("s", None, "future_maze_game", "a"))

    assert isinstance(decision, WithReasoning)
    assert isinstance(decision.action, LegalAction) and decision.action.action_id == "east"
    assert decision.reasoning_summary == "East reaches the goal fastest."
    schema = provider.calls[0]["schema"]
    assert schema["properties"]["action_id"]["enum"] == ["north", "east"]
    assert schema["additionalProperties"] is False and set(schema["required"]) == {"action_id", "reasoning_summary"}
    prompt = provider.prompt()
    assert prompt["game_type"] == "future_maze_game"
    assert prompt["legal_actions"] == [{"action_id": "north", "label": "Move north"},
                                       {"action_id": "east", "label": "Move east"}]
    assert "authoritative" in provider.calls[0]["messages"][0]["content"]


def test_prompt_state_omits_transport_keys_and_is_capped(monkeypatch):
    monkeypatch.setattr(llm_agent, "STATE_CHAR_LIMIT", 200)
    provider = FakeProvider(_answer(action_id="x"))
    state = _state({"action_id": "x", "label": "x"}, observation={"big": "y" * 5000})

    _agent(provider).choose_action(state, POKEMON)

    text = provider.prompt()["state"]
    assert text.endswith("...(truncated)") and len(text) <= 200 + len("...(truncated)")
    assert '"legal_actions"' not in text


def test_malformed_output_retries_with_the_error_fed_back():
    log: list[str] = []
    provider = FakeProvider(ProviderError("the model's reply was not valid JSON"), _answer(action_id="a"))

    decision = _agent(provider, log).choose_action(_state({"action_id": "a", "label": "A"}), POKEMON)

    assert decision.action.action_id == "a"
    assert "not valid JSON" in provider.calls[1]["messages"][-1]["content"]
    assert any("attempt 1/2" in line for line in log)


def test_illegal_choice_retries():
    provider = FakeProvider(_answer(action_id="nope"), _answer(action_id="b"))

    decision = _agent(provider).choose_action(_state({"action_id": "a"}, {"action_id": "b"}), POKEMON)

    assert decision.action.action_id == "b"
    assert "not one of the legal action_ids" in provider.calls[1]["messages"][-1]["content"]


def test_retry_exhaustion_falls_back_to_first_legal_action_and_says_so():
    log: list[str] = []
    provider = FakeProvider(_answer(action_id="nope"), ProviderError("OpenAI request failed (HTTP 500)"))

    decision = _agent(provider, log).choose_action(_state({"action_id": "a"}, {"action_id": "b"}), POKEMON)

    assert len(provider.calls) == llm_agent.MAX_ATTEMPTS
    assert decision.action.action_id == "a"
    assert decision.reasoning_summary.startswith("Fallback")
    assert any("FALLBACK" in line for line in log)


def test_reasoning_summary_is_trimmed_and_capped():
    provider = FakeProvider(_answer(action_id="a", reasoning_summary="  " + "x" * 1000 + "  "))

    decision = _agent(provider).choose_action(_state({"action_id": "a"}), POKEMON)

    assert decision.reasoning_summary == "x" * llm_agent.REASONING_CHAR_LIMIT


def test_unknown_structured_template_fails_safely_without_calling_the_model():
    provider = FakeProvider()
    legal = _templated("submit_plan", {"type": "submit_plan", "plan": "Pass a list of steps", "instructions": "..."})

    with pytest.raises(UnsupportedStructuredAction, match="submit_plan"):
        _agent(provider).choose_action(_state(legal), DecisionContext("s", None, "future_planning_game", "a"))
    assert provider.calls == []


def test_unknown_structured_template_fails_the_match_through_the_runner():
    game = ScriptedGame([_raw_state(_templated("submit_team", {"type": "submit_team", "team": "Pass a list"}))], RESULT)

    with pytest.raises(DecisionError, match="submit_team"):
        run_game(game, POKEMON, _agent(FakeProvider()))
    assert game.play_calls == []


def test_template_detection_treats_concrete_payloads_as_ordinary_actions():
    # A draft pick's input.action is a finished move, not a template.
    provider = FakeProvider(_answer(action_id="draft_pick:b"))

    decision = _agent(provider).choose_action(_state(*_draft("a", "b"), phase="draft"), POKEMON)

    assert decision.action.action_id == "draft_pick:b"
    assert provider.calls[0]["schema_name"] == "choose_action"


def test_a_new_structured_game_only_needs_an_adapter():
    def plan_adapter(action, state):
        steps = action.input["action"]["steps"]

        def build(answer):
            if answer.get("step") not in steps:
                raise InvalidChoice("unknown step")
            return {"type": "submit_plan", "step": answer["step"]}

        return Choice("submit_plan", {"steps": steps}, object_schema({"step": {"type": "string", "enum": steps}}),
                      build, lambda: {"type": "submit_plan", "step": steps[0]})

    provider = FakeProvider(_answer(step="scout"))
    legal = _templated("submit_plan", {"type": "submit_plan", "steps": ["build", "scout"], "instructions": "..."})

    decision = _agent(provider, adapters={"submit_plan": plan_adapter}).choose_action(_state(legal), POKEMON)

    assert decision.action == {"type": "submit_plan", "step": "scout"}


def test_no_legal_actions_is_a_clear_error():
    with pytest.raises(InvalidChoice):
        _agent(FakeProvider()).choose_action(GameState.from_mcp_state({"session_id": "s"}), POKEMON)


# -- Pokémon ---------------------------------------------------------------------------------


def test_pokemon_draft_pick():
    observation = {"phase": "draft", "available_cards": [{"card_id": "b", "species": "Incineroar"}], "rosters": {}}
    provider = FakeProvider(_answer(action_id="draft_pick:b", reasoning_summary="Incineroar for Intimidate."))

    decision = _agent(provider).choose_action(_state(*_draft("a", "b"), phase="draft", observation=observation), POKEMON)

    assert decision.action.action_id == "draft_pick:b"
    assert decision.reasoning_summary == "Incineroar for Intimidate."
    prompt = provider.prompt()
    assert prompt["phase"] == "draft" and "Incineroar" in prompt["state"]


def test_pokemon_lineup_uses_model_bring_and_leads():
    provider = FakeProvider(_answer(bring=["Urshifu", "Amoonguss", "Incineroar", "Tornadus"], leads=["Tornadus", "Urshifu"]))

    decision = _agent(provider).choose_action(_state(_lineup(), phase="team_preview"), POKEMON)

    assert decision.action == {"type": "select_lineup", "bring": ["Urshifu", "Amoonguss", "Incineroar", "Tornadus"],
                               "leads": ["Tornadus", "Urshifu"]}
    assert provider.calls[0]["schema"]["properties"]["bring"]["items"]["enum"] == ROSTER
    assert provider.prompt()["roster"] == ROSTER


@pytest.mark.parametrize(
    "bring, leads",
    [
        (["Urshifu", "Amoonguss", "Incineroar"], ["Urshifu", "Amoonguss"]),
        (["Urshifu", "Urshifu", "Incineroar", "Tornadus"], ["Urshifu", "Tornadus"]),
        (["Urshifu", "Amoonguss", "Incineroar", "Mew"], ["Urshifu", "Amoonguss"]),
        (["Urshifu", "Amoonguss", "Incineroar", "Tornadus"], ["Rillaboom", "Urshifu"]),
    ],
)
def test_pokemon_invalid_lineup_is_retried(bring, leads):
    provider = FakeProvider(_answer(bring=bring, leads=leads), _answer(bring=ROSTER[:4], leads=ROSTER[:2]))

    decision = _agent(provider).choose_action(_state(_lineup(), phase="team_preview"), POKEMON)

    assert decision.action["bring"] == ROSTER[:4]
    assert len(provider.calls) == 2


def test_pokemon_normal_doubles_move():
    provider = FakeProvider(_answer(slot_0={"option": 1, "target": 2}, slot_1={"option": 0, "target": 1}))
    legal = _doubles([_move("fakeout", [1, 2]), _move("flareblitz", [1, 2]), _switch("Amoonguss")],
                     [_move("grassyglide", [1, 2]), _switch("Amoonguss")])

    decision = _agent(provider).choose_action(_state(legal), POKEMON)

    assert decision.action == {
        "type": "doubles_turn",
        "slot_0": {"type": "move", "move_id": "flareblitz", "target": 2},
        "slot_1": {"type": "move", "move_id": "grassyglide", "target": 1},
    }
    assert provider.prompt()["slots"][0]["options"][1]["move_id"] == "flareblitz"


def test_pokemon_target_must_be_legal_and_targetless_moves_omit_it():
    provider = FakeProvider(
        _answer(slot_0={"option": 0, "target": 2}, slot_1={"option": 0, "target": None}),
        _answer(slot_0={"option": 0, "target": -2}, slot_1={"option": 0, "target": None}),
    )

    decision = _agent(provider).choose_action(_state(_doubles([_move("pollenpuff", [-2, 1])], [_move("protect", [])])), POKEMON)

    assert decision.action["slot_0"] == {"type": "move", "move_id": "pollenpuff", "target": -2}
    assert decision.action["slot_1"] == {"type": "move", "move_id": "protect"}
    assert "not legal for pollenpuff" in provider.calls[1]["messages"][-1]["content"]


def test_pokemon_forced_switch():
    provider = FakeProvider(_answer(slot_0={"option": 1, "target": None}, slot_1={"option": 0, "target": None}))

    decision = _agent(provider).choose_action(
        _state(_doubles([_switch("Amoonguss"), _switch("Tornadus")], [PASS], forced=(True, False))), POKEMON
    )

    assert decision.action["slot_0"] == {"type": "switch", "species": "Tornadus"}
    assert decision.action["slot_1"] == PASS


def test_pokemon_double_switch_conflict_is_retried():
    options = [_switch("Amoonguss"), _switch("Tornadus")]
    provider = FakeProvider(
        _answer(slot_0={"option": 0, "target": None}, slot_1={"option": 0, "target": None}),
        _answer(slot_0={"option": 0, "target": None}, slot_1={"option": 1, "target": None}),
    )

    decision = _agent(provider).choose_action(_state(_doubles(options, list(options), forced=(True, True))), POKEMON)

    assert decision.action["slot_1"] == {"type": "switch", "species": "Tornadus"}
    assert "same Pokémon" in provider.calls[1]["messages"][-1]["content"]


def test_pokemon_pass_only_slot_and_both_pass_rule():
    provider = FakeProvider(
        _answer(slot_0={"option": 1, "target": None}, slot_1={"option": 1, "target": None}),  # both pass: invalid
        _answer(slot_0={"option": 0, "target": None}, slot_1={"option": 1, "target": None}),
    )

    decision = _agent(provider).choose_action(
        _state(_doubles([_switch("Amoonguss"), PASS], [_switch("Amoonguss"), PASS], forced=(True, True))), POKEMON
    )

    assert decision.action["slot_0"] == {"type": "switch", "species": "Amoonguss"}
    assert decision.action["slot_1"] == PASS


def test_pokemon_retry_exhaustion_uses_smoke_agent_fallback():
    legal = _doubles([_move("fakeout", [-2, 1]), _move("protect", [])], [_switch("Amoonguss"), PASS])
    bad = _answer(slot_0={"option": 9, "target": None}, slot_1={"option": 0, "target": None})

    decision = _agent(FakeProvider(bad, dict(bad))).choose_action(_state(legal), POKEMON)

    assert decision.action == smoke_agent.choose_action(_state(legal), POKEMON)
    assert decision.reasoning_summary.startswith("Fallback")


def test_pokemon_malformed_template_raises_before_any_model_call():
    provider = FakeProvider()

    with pytest.raises(smoke_agent.SmokeAgentError):
        _agent(provider).choose_action(_state(_templated("doubles_turn", {"type": "doubles_turn", "slots": []})), POKEMON)
    assert provider.calls == []


def test_pokemon_full_loop_propagates_reasoning_and_skips_waits():
    provider = FakeProvider(
        _answer(action_id="draft_pick:b", reasoning_summary="Took Incineroar for Intimidate."),
        _answer(slot_0={"option": 0, "target": 1}, slot_1={"option": 0, "target": None},
                reasoning_summary="Fake Out their lead while Rillaboom sets up."),
    )
    game = ScriptedGame(
        [
            _raw_state(phase="draft", my_turn=False, version=1),
            _raw_state(*_draft("a", "b"), phase="draft", version=2),
            _raw_state(phase="moving", my_turn=False, version=3),
            _raw_state(_doubles([_move("fakeout", [1, 2])], [PASS]), version=4),
            _raw_state(phase="moving", my_turn=False, version=5),
            _raw_state(phase="moving", my_turn=False, version=6, terminal=True),
        ],
        RESULT,
    )

    final = run_game(game, POKEMON, _agent(provider))

    assert final.is_terminal
    assert len(provider.calls) == 2 and game.waits == 3
    assert game.play_calls[0]["action_id"] == "draft_pick:b"
    assert game.play_calls[0]["reasoning_summary"] == "Took Incineroar for Intimidate."
    assert game.play_calls[1]["action"]["type"] == "doubles_turn"
    assert game.play_calls[1]["reasoning_summary"] == "Fake Out their lead while Rillaboom sets up."


# -- Werewolf ---------------------------------------------------------------------------------


def test_werewolf_night_action():
    provider = FakeProvider(_answer(action_id="3", reasoning_summary="I checked Player3 because they were quiet."))
    state = _ww(*_ww_actions(("0", "Investigate Player0"), ("3", "Investigate Player3")), sub_phase="night_seer")

    decision = _agent(provider).choose_action(state, WEREWOLF)

    assert decision.action.action_id == "3"
    prompt = provider.prompt()
    assert "night_seer" in prompt["state"] and "Your role: villager" in prompt["state"]
    assert provider.calls[0]["schema"]["properties"]["action_id"]["enum"] == ["0", "3"]


def test_werewolf_day_vote_including_abstain_option():
    provider = FakeProvider(_answer(action_id="4", reasoning_summary="I voted for Player4; their votes are suspicious."))
    state = _ww(*_ww_actions(("0", "Vote to lynch Player0"), ("4", "Vote to lynch Player4"), ("7", "Abstain")))

    decision = _agent(provider).choose_action(state, WEREWOLF)

    play = _validate_decision(decision, state.legal_actions)
    assert (play.action_id, play.reasoning_summary) == ("4", "I voted for Player4; their votes are suspicious.")


def test_werewolf_message_send_broadcast_and_private():
    provider = FakeProvider(
        {"decision": "send", "message": "Player4 is lying.", "recipient": None},
        {"decision": "send", "message": "Let's vote 4.", "recipient": 5},
    )
    agent = _agent(provider)

    first = agent.choose_message(_ww(phase="messaging", messages=[(0, 4, "I am the seer.")]), WEREWOLF)
    second = agent.choose_message(_ww(phase="messaging"), WEREWOLF)

    assert isinstance(first, SendMessage) and first.content == "Player4 is lying." and first.recipients == []
    assert second.recipients == [5]
    schema = provider.calls[0]["schema"]
    assert "reasoning_summary" not in schema["properties"]  # chat and public reasoning are separate
    prompt = provider.prompt()
    assert prompt["recent_messages"] == [{"from": "Player4", "to": "everyone", "text": "I am the seer."}]
    assert prompt["decision"] == "message"


def test_werewolf_message_invalid_recipient_retries_and_long_text_is_capped():
    provider = FakeProvider(
        {"decision": "send", "message": "hi", "recipient": 2},  # self
        {"decision": "send", "message": "word " * 100, "recipient": None},
    )

    message = _agent(provider).choose_message(_ww(phase="messaging"), WEREWOLF)

    assert len(message.content.split()) == llm_agent.MESSAGE_WORD_LIMIT
    assert "recipient must be" in provider.calls[1]["messages"][-1]["content"]


def test_werewolf_discussion_round_is_bounded_and_ends():
    provider = FakeProvider(
        {"decision": "send", "message": "one", "recipient": None},
        {"decision": "send", "message": "two", "recipient": None},
    )
    agent = _agent(provider)
    state = _ww(phase="messaging", version=5)

    replies = [agent.choose_message(state, WEREWOLF) for _ in range(4)]

    assert [r.content for r in replies[:2]] == ["one", "two"]
    assert replies[2] is TERMINATE_MESSAGING and replies[3] is TERMINATE_MESSAGING
    assert agent.message_calls == llm_agent.MESSAGE_REQUESTS_PER_ROUND == 2


def test_werewolf_model_ending_the_round_stops_further_message_calls_until_next_round():
    provider = FakeProvider({"decision": "end", "message": "", "recipient": None},
                            {"decision": "end", "message": "", "recipient": None})
    agent = _agent(provider)

    assert agent.choose_message(_ww(phase="messaging", version=5), WEREWOLF) is TERMINATE_MESSAGING
    assert agent.choose_message(_ww(phase="messaging", version=5), WEREWOLF) is TERMINATE_MESSAGING
    assert agent.message_calls == 1
    agent.choose_message(_ww(phase="messaging", version=9), WEREWOLF)  # next day's round
    assert agent.message_calls == 2


def test_werewolf_message_fallback_ends_the_round():
    log: list[str] = []
    provider = FakeProvider({"decision": "shout"}, ProviderError("OpenAI request failed (HTTP 500)"))

    assert _agent(provider, log).choose_message(_ww(phase="messaging"), WEREWOLF) is TERMINATE_MESSAGING
    assert any("FALLBACK" in line for line in log)


def test_werewolf_full_loop():
    """Wait -> night action -> discussion (send, end, auto-end) -> day vote -> wait -> terminal."""
    provider = FakeProvider(
        _answer(action_id="3", reasoning_summary="Night: I chose Player3."),
        {"decision": "send", "message": "I think Player4 is a wolf.", "recipient": None},
        {"decision": "end", "message": "", "recipient": None},
        _answer(action_id="4", reasoning_summary="I voted for Player4 because of their claims."),
    )
    night = _ww_actions(("0", "Kill Player0"), ("3", "Kill Player3"))
    vote = _ww_actions(("0", "Vote to lynch Player0"), ("4", "Vote to lynch Player4"), ("7", "Abstain"))
    game = ScriptedGame(
        [
            _ww_raw(my_turn=False, sub_phase="night_wolf", version=1),                        # get_state: waiting
            _ww_raw(*night, sub_phase="night_wolf", version=1),                                # wait -> my night move
            _ww_raw(phase="messaging", my_turn=False, version=2, messages=[(0, 5, "Morning.")]),  # after move
            _ww_raw(phase="messaging", my_turn=False, version=2, messages=[(1, 2, "I think..."), (2, 4, "Not me!")]),
            _ww_raw(phase="messaging", my_turn=False, version=2, messages=[(3, 6, "Vote soon.")]),
            _ww_raw(*vote, version=2),                                                         # get_state: vote
            _ww_raw(my_turn=False, version=3),                                                 # after vote
            _ww_raw(my_turn=False, version=4, terminal=True),                                  # wait -> over
        ],
        RESULT,
        send_results=[{"phase": "messaging"}, {"phase": "messaging"}, {"phase": "moving"}],
    )
    agent = _agent(provider)

    final = run_game(game, WEREWOLF, agent)

    assert final.is_terminal
    assert (agent.decision_calls, agent.message_calls) == (2, 2)
    assert provider.answers == []
    assert [c["action_id"] for c in game.play_calls] == ["3", "4"]
    assert game.play_calls[1]["reasoning_summary"] == "I voted for Player4 because of their claims."
    assert [c["message_type"] for c in game.send_calls] == ["chat", "terminate", "terminate"]
    assert game.send_calls[0]["content"] == "I think Player4 is a wolf."
    assert game.send_calls[0]["content"] != game.play_calls[1]["reasoning_summary"]
    vote_prompt = provider.prompt(3)
    assert {"from": "Player4", "to": "everyone", "text": "Not me!"} in vote_prompt["recent_messages"]


def test_no_model_call_while_waiting_or_after_terminal():
    provider = FakeProvider()
    game = ScriptedGame(
        [_ww_raw(my_turn=False, version=1), _ww_raw(my_turn=False, version=1), _ww_raw(my_turn=False, version=2, terminal=True)],
        RESULT,
    )

    run_game(game, WEREWOLF, _agent(provider))

    assert provider.calls == [] and game.waits == 2


# -- reasoning plumbing -------------------------------------------------------------------------


def test_with_reasoning_validates_inner_action_and_rejects_resign():
    legal = _state(*_draft("a")).legal_actions

    play = _validate_decision(WithReasoning(legal[0], "Good pick."), legal)

    assert (play.action_id, play.reasoning_summary) == ("draft_pick:a", "Good pick.")
    with pytest.raises(DecisionError):
        _validate_decision(WithReasoning(RESIGN, "x"), legal)
    with pytest.raises(DecisionError):
        _validate_decision(WithReasoning("draft_pick:nope", "x"), legal)


def test_mcp_play_action_sends_reasoning_summary_only_when_given(monkeypatch):
    import altruagent.mcp_game as mcp_game_module

    calls = []
    monkeypatch.setattr(mcp_game_module, "call_tool", lambda client, url, tool, args: calls.append(args) or {})
    game = MCPGameSession(object(), session_id="s", game_server_url="https://g.example.test")

    game.play_action(action_id="x", state_version=1)
    game.play_action(action={"type": "t"}, state_version=2, reasoning_summary="Because.")

    assert "reasoning_summary" not in calls[0]
    assert calls[1] == {"session_id": "s", "state_version": 2, "action": {"type": "t"}, "reasoning_summary": "Because."}


# -- provider -----------------------------------------------------------------------------------------


def _openai(handler, **kwargs):
    return OpenAIProvider(FAKE_KEY, "gpt-test", base_url="https://openai.example.test/v1",
                          transport=httpx.MockTransport(handler), **kwargs)


def test_openai_request_uses_strict_json_schema_bearer_key_and_base_url():
    seen = {}

    def handler(request):
        seen.update(auth=request.headers["authorization"], url=str(request.url), body=json.loads(request.content))
        content = json.dumps({"action_id": "a", "reasoning_summary": "ok"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    answer = _openai(handler).complete_structured([{"role": "user", "content": "hi"}], "choose_action", {"type": "object"})

    assert answer == {"action_id": "a", "reasoning_summary": "ok"}
    assert seen["auth"] == f"Bearer {FAKE_KEY}"
    assert seen["url"] == "https://openai.example.test/v1/chat/completions"
    assert seen["body"]["model"] == "gpt-test"
    assert seen["body"]["response_format"]["json_schema"] == {"name": "choose_action", "strict": True,
                                                             "schema": {"type": "object"}}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "[1, 2]"}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": None, "refusal": "no"}}]}),
        httpx.Response(200, json={"unexpected": True}),
    ],
)
def test_openai_malformed_replies_raise_provider_error(response):
    with pytest.raises(ProviderError):
        _openai(lambda request: response).complete_structured([], "x", {})


def test_api_key_never_appears_in_errors_logs_or_repr(capsys):
    def handler(request):
        # Real providers can echo part of the key in an auth error body.
        return httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {FAKE_KEY}"}})

    provider = _openai(handler)
    agent = LLMAgent(provider)  # default log=print

    decision = agent.choose_action(_state({"action_id": "a"}), POKEMON)
    agent.choose_message(_ww(phase="messaging"), WEREWOLF)

    out = capsys.readouterr().out
    assert "HTTP 401" in out and "FALLBACK" in out
    assert FAKE_KEY not in out
    assert FAKE_KEY not in repr(provider) and FAKE_KEY not in repr(agent) and FAKE_KEY not in repr(decision)
    with pytest.raises(ProviderError) as exc_info:
        provider.complete_structured([], "x", {})
    assert FAKE_KEY not in str(exc_info.value)


def test_network_error_message_hides_key():
    def handler(request):
        raise httpx.ConnectError(f"failed for {FAKE_KEY}")

    with pytest.raises(ProviderError) as exc_info:
        _openai(handler).complete_structured([], "x", {})

    assert FAKE_KEY not in str(exc_info.value) and exc_info.value.__cause__ is None


def test_provider_from_env_configures_model_and_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-custom")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.test/v1/")

    provider = provider_from_env()

    assert provider.model == "gpt-custom"
    assert str(provider._http.base_url) == "https://proxy.example.test/v1/"


def test_provider_from_env_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    provider = provider_from_env()

    assert provider.model == providers.DEFAULT_OPENAI_MODEL == "gpt-4o-mini"
    assert str(provider._http.base_url).startswith(providers.DEFAULT_OPENAI_BASE_URL)


def test_create_agent_without_key_fails_clearly(monkeypatch):
    monkeypatch.setattr(llm_agent, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set"):
        create_agent()


def test_create_agent_builds_an_llm_agent_with_the_pokemon_adapters(monkeypatch):
    monkeypatch.setattr(llm_agent, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)

    agent = create_agent()

    assert isinstance(agent, LLMAgent)
    assert set(llm_agent.STRUCTURED_ADAPTERS) == {"select_lineup", "doubles_turn"}


def _named_move(move_id, target_options):
    move = _move(move_id, [t["target"] for t in target_options])
    move["target_options"] = target_options
    return move


def test_pokemon_prompt_names_every_target_and_sends_the_integer_back():
    # The real turn-4 board: own [Dondozo, Chien-Pao] vs [Hatterene, Cresselia].
    # The model once attacked its own Dondozo reading -1 as a foe.
    wave_crash = _named_move("wavecrash", [
        {"target": -2, "side": "ally", "species": "chienpao"},
        {"target": 1, "side": "opponent", "species": "hatterene"},
        {"target": 2, "side": "opponent", "species": "cresselia"},
    ])
    sacred_sword = _named_move("sacredsword", [
        {"target": -1, "side": "ally", "species": "dondozo"},
        {"target": 1, "side": "opponent", "species": "hatterene"},
        {"target": 2, "side": "opponent", "species": "cresselia"},
    ])
    provider = FakeProvider(_answer(slot_0={"option": 0, "target": 1}, slot_1={"option": 0, "target": -1}))

    decision = _agent(provider).choose_action(_state(_doubles([wave_crash], [sacred_sword])), POKEMON)

    prompt = provider.prompt()
    slot_1_targets = prompt["slots"][1]["options"][0]["targets"]
    assert slot_1_targets == [
        {"target": -1, "is": "ALLY dondozo — your OTHER active Pokémon (legal, but it hits your own side)"},
        {"target": 1, "is": "OPPONENT hatterene"},
        {"target": 2, "is": "OPPONENT cresselia"},
    ]
    assert "target_options" not in prompt["slots"][1]["options"][0]
    for word in ("SELF", "ALLY", "OPPONENT"):
        assert word in prompt["instructions"]
    # Ally targeting stays legal, and GameAPI still receives the plain integer.
    assert decision.action["slot_0"] == {"type": "move", "move_id": "wavecrash", "target": 1}
    assert decision.action["slot_1"] == {"type": "move", "move_id": "sacredsword", "target": -1}


def test_pokemon_self_and_targetless_moves_are_named_too():
    acupressure = _named_move("acupressure", [
        {"target": -1, "side": "self", "species": "dondozo"},
        {"target": -2, "side": "ally", "species": "chienpao"},
    ])
    protect = _named_move("protect", [{"target": 0, "side": "none", "species": None}])
    provider = FakeProvider(_answer(slot_0={"option": 0, "target": -1}, slot_1={"option": 0, "target": 0}))

    _agent(provider).choose_action(_state(_doubles([acupressure], [protect])), POKEMON)

    slots = provider.prompt()["slots"]
    assert slots[0]["options"][0]["targets"][0]["is"] == "SELF dondozo — this Pokémon itself"
    assert slots[1]["options"][0]["targets"] == [
        {"target": 0, "is": "no target needed (self, field or spread move)"}
    ]


def test_pokemon_prompt_without_target_options_keeps_the_raw_integers():
    # An older GameAPI that sends no target_options: unchanged behaviour.
    provider = FakeProvider(_answer(slot_0={"option": 0, "target": 2}, slot_1={"option": 0, "target": None}))

    _agent(provider).choose_action(_state(_doubles([_move("fakeout", [1, 2])], [PASS])), POKEMON)

    assert provider.prompt()["slots"][0]["options"][0]["targets"] == [1, 2]


# -- Pokémon's clocks: 55 s per battle decision, 90 s at Team Preview, 15 s per draft pick ------


class TimedProvider:
    """A provider that takes ``timeout`` and spends scripted seconds on a fake clock."""

    model = "fake-model"

    def __init__(self, clock, *steps, honors_timeout=True):
        self.clock = clock
        self.steps = list(steps)  # (seconds spent, answer or exception)
        self.honors_timeout = honors_timeout
        self.timeouts: list[float | None] = []

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.timeouts.append(timeout)
        spent, answer = self.steps.pop(0)
        if self.honors_timeout and timeout is not None and spent > timeout:
            self.clock["t"] += timeout
            raise ProviderError("OpenAI request failed (ReadTimeout)")
        self.clock["t"] += spent
        if isinstance(answer, Exception):
            raise answer
        return answer


def _timed_agent(provider, clock, log=None):
    return LLMAgent(provider, log=(log if log is not None else []).append, clock=lambda: clock["t"])


def test_a_pokemon_battle_decision_never_waits_longer_than_its_budget_even_when_the_model_hangs():
    clock = {"t": 0.0}
    legal = _doubles([_move("fakeout", [-2, 1]), _move("protect", [])], [_switch("Amoonguss"), PASS])
    provider = TimedProvider(clock, (999, None), (999, None))
    log: list[str] = []

    decision = _timed_agent(provider, clock, log).choose_action(_state(legal), POKEMON)

    assert decision.action == smoke_agent.choose_action(_state(legal), POKEMON)  # the fallback, in time
    assert clock["t"] <= llm_agent.POKEMON_DECISION_SECONDS < 45  # Showdown plays a default move at 55 s
    assert provider.timeouts == [llm_agent.POKEMON_REQUEST_SECONDS,
                                 llm_agent.POKEMON_DECISION_SECONDS - llm_agent.POKEMON_REQUEST_SECONDS]
    assert any("FALLBACK" in line for line in log)


def test_a_slow_invalid_first_answer_leaves_the_retry_only_the_time_that_is_left():
    clock = {"t": 0.0}
    provider = TimedProvider(clock, (20, _answer(action_id="nope")), (5, _answer(action_id="b")))

    decision = _timed_agent(provider, clock).choose_action(
        _state({"action_id": "a"}, {"action_id": "b"}), POKEMON)

    assert decision.action.action_id == "b"
    assert provider.timeouts == [25.0, 20.0]


def test_no_retry_is_started_with_almost_no_time_left():
    clock = {"t": 0.0}
    # A request that overran its timeout (slow DNS, a proxy) leaves 2 s: too little to try again.
    provider = TimedProvider(clock, (38, _answer(action_id="nope")), honors_timeout=False)
    log: list[str] = []

    decision = _timed_agent(provider, clock, log).choose_action(
        _state({"action_id": "a"}, {"action_id": "b"}), POKEMON)

    assert decision.action.action_id == "a" and len(provider.timeouts) == 1
    assert any("out of time" in line for line in log)


def test_team_preview_fits_its_90_seconds():
    clock = {"t": 0.0}
    provider = TimedProvider(clock, (999, None), (999, None))

    decision = _timed_agent(provider, clock).choose_action(_state(_lineup(), phase="team_preview"), POKEMON)

    assert decision.action["type"] == "select_lineup"
    assert clock["t"] <= llm_agent.POKEMON_DECISION_SECONDS < 90


def test_a_draft_pick_fits_its_15_seconds():
    clock = {"t": 0.0}
    provider = TimedProvider(clock, (999, None), (999, None))

    decision = _timed_agent(provider, clock).choose_action(_state(*_draft("a", "b"), phase="draft"), POKEMON)

    assert decision.action.action_id == "draft_pick:a"
    assert clock["t"] <= llm_agent.POKEMON_DRAFT_SECONDS < 15
    assert provider.timeouts[0] == llm_agent.POKEMON_DRAFT_SECONDS


def test_werewolf_keeps_its_old_timing():
    clock = {"t": 0.0}
    provider = TimedProvider(clock, (70, _answer(action_id="1")))

    decision = _timed_agent(provider, clock).choose_action(_ww(*_ww_actions(("1", "Player1"))), WEREWOLF)

    assert decision.action.action_id == "1" and provider.timeouts == [None]


def test_a_provider_without_a_timeout_parameter_still_works_and_only_its_attempts_are_limited():
    clock = {"t": 0.0}

    class OldProvider:
        model = "old"

        def __init__(self):
            self.calls = 0

        def complete_structured(self, messages, schema_name, schema):
            self.calls += 1
            clock["t"] += 39
            return _answer(action_id="nope")

    provider = OldProvider()
    decision = _timed_agent(provider, clock).choose_action(_state({"action_id": "a"}), POKEMON)

    assert decision.action.action_id == "a" and provider.calls == 1


def test_openai_provider_applies_a_per_request_timeout_only_when_asked():
    seen = []

    def handler(request):
        seen.append(request.extensions.get("timeout"))
        content = json.dumps({"action_id": "a", "reasoning_summary": "ok"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = _openai(handler)
    provider.complete_structured([], "x", {})
    provider.complete_structured([], "x", {}, timeout=12.5)

    assert seen[0]["read"] == providers.REQUEST_TIMEOUT_SECONDS
    assert seen[1] == {"connect": 12.5, "read": 12.5, "write": 12.5, "pool": 12.5}


def test_the_real_openai_provider_is_given_the_pokemon_timeout():
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"]["read"])
        content = json.dumps({"action_id": "a", "reasoning_summary": "ok"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    LLMAgent(_openai(handler), log=lambda line: None).choose_action(_state({"action_id": "a"}), POKEMON)
    LLMAgent(_openai(handler), log=lambda line: None).choose_action(_ww(*_ww_actions(("1", "Player1"))), WEREWOLF)

    assert seen[0] <= llm_agent.POKEMON_REQUEST_SECONDS
    assert seen[1] == providers.REQUEST_TIMEOUT_SECONDS
