

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
