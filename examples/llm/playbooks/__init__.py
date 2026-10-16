"""Per-game strategy playbooks for the example LLM agent.

``<game_type>.md`` in this folder is added to the model's instructions for
that game (Pokémon: every decision; Red Alert: the standing system prompt).
Edit the Markdown to change strategy; no code changes needed. A game with no
file gets no playbook; one listed in ``OFF_BY_DEFAULT`` is used only when its environment
switch is "on" (Pokémon: ``POKEMON_PLAYBOOK=on``). ``ENABLED = False`` turns them off for this process
(``examples/mixed_table.py`` uses it for opponent seats).
"""

from __future__ import annotations

import os
from pathlib import Path

ENABLED = True
# Playbooks that are off unless their env switch says "on". Pokémon: in 12 test games of
# GPT-6.1-sol with this playbook against GPT-6.1-sol without it, the playbook side won 3.
OFF_BY_DEFAULT = {"pokemon_vgc_doubles_draft": "POKEMON_PLAYBOOK"}
_FOLDER = Path(__file__).parent


def playbook(game_type: str | None) -> str:
    """The playbook text for ``game_type``, or "" (no file, or playbooks off)."""
    if not ENABLED or not game_type:
        return ""
    switch = OFF_BY_DEFAULT.get(game_type)
    if switch and (os.environ.get(switch) or "").lower() not in ("on", "1", "true", "yes"):
        return ""
    path = _FOLDER / f"{game_type}.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
