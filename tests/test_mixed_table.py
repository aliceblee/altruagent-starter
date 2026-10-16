

def test_dummy_seats_get_their_own_team_names_everywhere():
    import examples.mixed_table as m
    from agent.werewolf.parsing import seat_teams, snapshot
    from tests.test_werewolf_agent import state

    s = state(0, "Your role: villager.", phase="messaging",
              votes=[{"day": 1, "votes": {"alime588 (Player3, Self-hosted)": "alime588 (Player4, Self-hosted)"}}],
              dead=[{"player": "alime588 (Player4, Self-hosted)", "role": "villager", "cause": "vote", "day": 1}])
    s.raw["players"] = [{"position": i, "name": f"alime588 (Player{i}, Self-hosted)"} for i in range(7)]
    m.Mixed(seats={0: "ours", 3: "claude", 4: "gemini"}, brains={})._relabel(s)
    assert seat_teams(s.raw)[0] == "alime588" and seat_teams(s.raw)[3] == "Test-claude"
    snap, _ = snapshot(s, 0)
    assert [(v.voter, v.target) for v in snap.votes] == [(3, 4)] and snap.dead[4].cause == "vote"


def test_two_player_games_use_their_own_seat_map_and_pass_structured_moves_through(monkeypatch):
    import examples.mixed_table as m
    from altruagent import DecisionContext, WAIT
    from altruagent.models import GameState

    made = []

    class Echo:
        def __init__(self, seat):
            made.append(seat)

        def choose_action(self, state, context):
            return {"type": "select_lineup", "lineup": [1, 2, 3, 4]}  # a structured payload, not a LegalAction

    monkeypatch.setattr(m, "SEATS_BY_GAME", {"pokemon_vgc_doubles_draft": {0: "ours", 1: "random"}})
    brains = {"ours": lambda seat: Echo(seat), m.UNLISTED_SEAT_OTHER_GAMES: lambda seat: Echo(seat)}
    state = GameState.from_mcp_state({"session_id": "p", "game_type": "pokemon_vgc_doubles_draft", "phase": "moving",
                                      "your_position": 1, "players": []})
    seat = m.Mixed(brains=brains)
    ctx = DecisionContext(session_id="p", tournament_id=None, game_type="pokemon_vgc_doubles_draft", agent_id="a", seat_position=1)
    assert seat.choose_action(state, ctx) == {"type": "select_lineup", "lineup": [1, 2, 3, 4]}
    assert seat.name == m.UNLISTED_SEAT_OTHER_GAMES  # "random" only plays Werewolf
    assert made == [1]


def test_comparison_seats_set_the_provider_for_their_own_process_only(monkeypatch):
    import os
    import examples.mixed_table as m

    built = []
    monkeypatch.setattr("agent.agent.Agent", lambda brain=None: built.append(dict(os.environ)) or object())
    monkeypatch.setattr(m, "load_dotenv", lambda: None)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(os, "environ", dict(os.environ))  # keep the test's own environment clean
    m.BRAINS["ours_openai"](1)
    assert built[-1]["LLM_PROVIDER"] == "openai" and built[-1]["OPENAI_MODEL"] == m.OPENAI_61_MODEL


def test_a_failing_llm_seat_abstains_in_werewolf_instead_of_voting_seat_0():
    import examples.mixed_table as m
    from altruagent import DecisionContext, WithReasoning
    from tests.test_werewolf_agent import state

    class Broken:
        def choose_action(self, st, context):
            return WithReasoning(st.legal_actions[0], "Fallback: the model gave no valid answer, so a default legal action was played.")

    seat = m.Mixed(seats={3: "broken"}, brains={"broken": lambda s: Broken()})
    ctx = DecisionContext(session_id="w", tournament_id=None, game_type="werewolf", agent_id="a", seat_position=3)
    move = seat.choose_action(state(3, "Your role: villager.", legal=[0, 1, 2, 4, 5, 6, 7]), ctx)
    assert m._legal(move, state(3, "x", legal=[0, 1, 2, 4, 5, 6, 7])).action_id == "7"
