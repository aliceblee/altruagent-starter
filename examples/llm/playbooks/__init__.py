"""Per-game strategy playbooks for the example LLM agent.

``<game_type>.md`` in this folder is added to the model's instructions for
that game (Pokémon: every decision; Red Alert: the standing system prompt).
Edit the Markdown to change strategy; no code changes needed. A game with no
file gets no playbook. ``ENABLED = False`` turns them off for this process
(``examples/mixed_table.py`` uses it for opponent seats).
"""

from __future__ import annotations

from pathlib import Path

ENABLED = True
_FOLDER = Path(__file__).parent


def playbook(game_type: str | None) -> str:
    """The playbook text for ``game_type``, or "" (no file, or playbooks off)."""
    if not ENABLED or not game_type:
        return ""
    path = _FOLDER / f"{game_type}.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
