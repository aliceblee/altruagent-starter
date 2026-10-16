import pytest


@pytest.fixture(autouse=True)
def _decision_logs_in_tmp(tmp_path, monkeypatch):
    """Keep agent/decision_log.py's per-match logs out of the repository."""
    monkeypatch.setenv("ALTRUAGENT_DECISION_LOG_DIR", str(tmp_path / "decisions"))
    # Tests never write game records or learned patterns into the repo.
    monkeypatch.setenv("ALTRUAGENT_GAME_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("WEREWOLF_AUTO_LEARN", "0")
    # A developer's real keys must not change which provider a test gets.
    for name in ("ANTHROPIC_API_KEY", "LLM_PROVIDER", "CLAUDE_MODEL", "CLAUDE_EFFORT", "CLAUDE_DEEP_MODEL", "CLAUDE_THINKING_EFFORT", "WEREWOLF_BRAIN"):
        monkeypatch.delenv(name, raising=False)
