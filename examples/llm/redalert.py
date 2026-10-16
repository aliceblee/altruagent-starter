"""Red Alert for the example LLM agent: the model decides each batch of orders.

Red Alert (``red_alert``) is real time: there are no turns, the world runs at
25 ticks per second whether or not you act, and a move is a batch of 1-20
orders (``{"type": "orders", "orders": [...]}``) sent whenever the agent is
ready. ``examples/llm_agent.py`` hands every Red Alert decision to a
``RedAlertPlayer``, one per match.

This is a port of the platform's own Red Alert test agent (Agent_ACP
``gameapi/scripts/redalert_llm_brain.py``), with the same prompts and the same
safeguards, measured in live matches:

1. The system prompt states the goal, the real-time rules, every order's fields
   (from ``context.game_config``'s ``order_schema`` when the runner fetched it)
   and the opening build order. The user prompt is a compact JSON view of this
   state (at most ~12,000 characters; lists are capped and the cut is noted).
2. The model answers through a strict JSON schema with one shape per order
   command, so every order is well formed; each order is still checked here
   (ids, cells, counts), and GameAPI checks everything again.
3. An empty batch, or any decision the model can't make (a provider error, an
   unusable answer, the call cap), sends nothing: the player returns
   ``altruagent.WAIT`` and is asked again on the next view. Real-time speed is
   part of the game, so a slow or failing model simply acts less. After a
   provider error the model is left alone for 5 s, doubling up to 60 s.
4. Refusals are fed back. ``on_action_result`` gets every ``play_action``
   answer: a batch refused as a whole (``INVALID_ACTION``, which never appears
   in ``last_orders``) is shown to the model with its reasons, and an order
   refused again and again for the same reason is listed, then dropped before
   sending while the reason still holds (up to 30 s at a time), so the model
   can't spend a match repeating one refused order.
5. Attacking is always among the suggestions: the server's ready-made attack
   (``legal_actions.attack_now``) is put in front of the model whenever the
   seat has combat units. Buildings queued several times are pointed out.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from altruagent import WAIT, DecisionContext, GameState, WithReasoning

from .base import object_schema
from .playbooks import playbook
from .providers import LLMProvider, ProviderError

# A cost guard sized for a whole 20-minute match at one answer every 0.5 s.
MAX_CALLS = 2400

REASONING_MAX_CHARS = 300
USER_PROMPT_MAX_CHARS = 12000
CONFIG_TEXT_MAX_CHARS = 400
MAX_UNITS_SHOWN = 60
MAX_BUILDINGS_SHOWN = 60
MAX_ENEMIES_SHOWN = 40
MAX_PRODUCTION_SHOWN = 20
MAX_AVAILABLE_SHOWN = 40
MAX_EVENTS_SHOWN = 20
MAX_ORDER_RESULTS_SHOWN = 20
MAX_PROBLEMS_SHOWN = 10
DETAIL_MAX_CHARS = 80

# Batches refused as a whole (play_action INVALID_ACTION: nothing was sent, and
# the batch never shows in last_orders or recent_order_problems). The newest are
# shown to the model so it can see it is repeating a refused move.
MAX_REFUSED_SHOWN = 3
REFUSED_DETAIL_MAX_CHARS = 300
REFUSED_NOTE = (
    "These batches of yours were refused entirely: none of their orders ran, and "
    "they do not appear in last_orders. Do not send the same orders again. Read "
    "each reason and choose different orders (other units, ids, items or cells)."
)
# Orders refused again and again, tracked by exact content and refusal reason.
# After REPEAT_WARN_AFTER identical refusals an order is listed in the prompt;
# after BLOCK_AFTER it is also dropped before sending while the reason still
# holds (checked against the state where possible) and for at most
# BLOCK_SECONDS since the last refusal, after which one more try goes out. A
# refusal older than REFUSAL_FORGET_SECONDS is forgotten.
REPEAT_WARN_AFTER = 2
BLOCK_AFTER = 3
BLOCK_SECONDS = 30.0
REFUSAL_FORGET_SECONDS = 120.0
MAX_REPEATS_SHOWN = 8
REPEATS_NOTE = (
    "Each of these orders was refused again and again for the same reason. Do not "
    "send it again until that reason no longer holds; send something else. Orders "
    "marked blocked are dropped before sending."
)
# The same building queued this many times or more, not yet ready.
OVERQUEUE_WARN_AT = 3
OVERQUEUE_NOTE = (
    "These buildings are already queued several times. Every extra build order "
    "queues another copy; send build for an item only when it is not queued."
)
SUGGESTED_ATTACK_NOTE = (
    "A ready attack order from the server: your combat units attack-move to the "
    "nearest enemy you can see, or to the enemy base. Games are won by destroying "
    "the enemy's buildings, so add it to your orders whenever you have a few combat "
    "units (you may change the units or the target)."
)
# A whole-batch refusal's detail: "...: [0] place: not_ready (spen ...); [1] ..."
_REFUSAL_ROW = re.compile(r"\[(\d+)\] ([a-z_]+): ([a-z_]+)(?: \(([^)]*)\))?")
# When the user prompt is too long: sections dropped first, then lists halved.
LOW_PRIORITY_SECTIONS = ("events", "your_last_reasoning", "recent_order_problems", "last_orders", "military", "limits")
SHRINKABLE_LISTS = ("enemies", "units", "buildings", "available_production", "production")
MIN_LIST_SHOWN = 5
MAX_ITEM_NAME = 32

# Fallback reasons (counted in stats()).
FALLBACK_CAP = "cap"
FALLBACK_ERROR = "error"
FALLBACK_BAD_ANSWER = "bad_answer"
FALLBACK_BACKOFF = "backoff"
BACKOFF_FIRST_SECONDS = 5.0
BACKOFF_MAX_SECONDS = 60.0

DEFAULT_LIMITS = {"max_orders_per_batch": 20, "max_units_per_order": 50, "max_train_count": 10}

MCV = "mcv"
CONSTRUCTION_YARD = "fact"
POWER_PLANT = "powr"
BARRACKS = ("tent", "barr")
REFINERY = "proc"
INFANTRY = "e1"

# cmd -> (required fields, optional fields). The allowed fields come from the
# game config's order_schema when the runner fetched it; the required ones from here.
ORDER_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "move": (frozenset({"units", "to"}), frozenset({"queued"})),
    "attack_move": (frozenset({"units", "to"}), frozenset({"queued"})),
    "attack": (frozenset({"units"}), frozenset({"target", "to", "queued"})),
    "stop": (frozenset({"units"}), frozenset()),
    "deploy": (frozenset({"units"}), frozenset()),
    "unload": (frozenset({"units"}), frozenset()),
    "guard": (frozenset({"units", "target"}), frozenset({"queued"})),
    "enter_transport": (frozenset({"units", "target"}), frozenset({"queued"})),
    "harvest": (frozenset({"units"}), frozenset({"to", "queued"})),
    "set_stance": (frozenset({"units", "stance"}), frozenset()),
    "build": (frozenset({"item"}), frozenset()),
    "train": (frozenset({"item"}), frozenset({"count"})),
    "place": (frozenset({"item"}), frozenset({"at"})),
    "cancel": (frozenset({"item"}), frozenset()),
    "sell": (frozenset({"building"}), frozenset()),
    "repair": (frozenset({"building"}), frozenset({"on"})),
    "power_toggle": (frozenset({"building"}), frozenset({"on"})),
    "set_primary": (frozenset({"building"}), frozenset()),
    "rally": (frozenset({"building", "to"}), frozenset()),
}
DEFAULT_STANCES = ("hold_fire", "return_fire", "defend", "attack_anything")

# JSON-schema types of the order fields, for the model's answer schema.
_INTS = {"type": "array", "items": {"type": "integer"}}
FIELD_SCHEMAS: dict[str, dict] = {
    "units": _INTS,
    "to": _INTS,
    "at": _INTS,
    "target": {"type": "integer"},
    "building": {"type": "integer"},
    "item": {"type": "string"},
    "count": {"type": "integer"},
    "queued": {"type": "boolean"},
    "stance": {"type": "string"},
    "on": {"type": "boolean"},
}

# One example per command family (ids and cells are placeholders).
ORDER_EXAMPLES = (
    {"cmd": "deploy", "units": [101]},
    {"cmd": "attack_move", "units": [120, 121, 122], "to": [50, 52]},
    {"cmd": "attack", "units": [120], "target": 305},
    {"cmd": "set_stance", "units": [120, 121], "stance": "defend"},
    {"cmd": "build", "item": "powr"},
    {"cmd": "place", "item": "powr"},
    {"cmd": "train", "item": "e1", "count": 3},
    {"cmd": "rally", "building": 130, "to": [20, 24]},
    {"cmd": "repair", "building": 131},
)

# Used only when the game config does not carry the text.
FALLBACK_WIN = (
    "You win by destroying every enemy building and the enemy's MCV (base "
    "vehicle); units left over do not matter. Or the enemy resigns."
)
FALLBACK_TIEBREAK = "higher kills_cost - deaths_cost wins; if equal, higher assets_value; if still equal, a draw"
FALLBACK_REALTIME = "The game runs in real time and never waits for you: while you think, the world keeps moving."
FALLBACK_MAP_TERMS = "bounds [x, y, w, h] = playable cells"


# --- Small helpers -------------------------------------------------------------


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any, fallback: str, limit: int = CONFIG_TEXT_MAX_CHARS) -> str:
    """A config string, whitespace-collapsed and trimmed; ``fallback`` if absent."""
    text = " ".join(value.split()) if isinstance(value, str) else ""
    return (text or fallback)[:limit]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_cell(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(map(_is_int, value))


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def trim_reasoning(value: Any) -> str | None:
    """The model's public reasoning, whitespace collapsed, at most 300 characters."""
    if not isinstance(value, str):
        return None
    return " ".join(value.split())[:REASONING_MAX_CHARS] or None


def backoff_seconds(errors_in_row: int) -> float:
    """How long to leave the model alone after provider errors in a row: 5, 10, 20, 40, then 60 s."""
    return min(BACKOFF_MAX_SECONDS, BACKOFF_FIRST_SECONDS * 2 ** min(max(0, errors_in_row - 1), 8))


def _cell(obj: Any) -> tuple[int, int] | None:
    """A cell from ``{"x", "y"}`` or ``[x, y]``."""
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        x, y = obj[0], obj[1]
    elif isinstance(obj, dict):
        x, y = obj.get("x"), obj.get("y")
    else:
        return None
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return int(x), int(y)
    return None


def _list(obs: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = obs.get(key)
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _limits(state: dict[str, Any]) -> dict[str, int]:
    limits = dict(DEFAULT_LIMITS)
    raw = state.get("limits")
    if isinstance(raw, dict):
        limits.update({k: v for k, v in raw.items() if _is_int(v) and v > 0})
    return limits


def _obs_tick(state: dict[str, Any]) -> int | None:
    tick = _dict(state.get("observation")).get("tick")
    return tick if _is_int(tick) else None


def find_map_info(config: dict[str, Any] | None, map_id: str | None) -> dict[str, Any] | None:
    """The map's entry (width/height/bounds/spawns) in ``get_game_config``'s ``rules.maps``."""
    entry = _dict(_dict(_dict(config).get("rules")).get("maps")).get(map_id or "")
    return entry if isinstance(entry, dict) else None


def _spawns(map_info: dict[str, Any] | None) -> list[tuple[int, int]]:
    cells = [_cell(s) for s in _dict(map_info).get("spawns") or []]
    return [c for c in cells if c is not None]


def choose_enemy_spawn(spawns: list[tuple[int, int]], base: tuple[int, int] | None) -> tuple[int, int] | None:
    """The spawn farthest from our base (ties: the first in map order)."""
    if not spawns or base is None:
        return None
    return max(spawns, key=lambda s: (s[0] - base[0]) ** 2 + (s[1] - base[1]) ** 2)


def playable_cells(bounds: Any) -> dict[str, list[int]] | None:
    """``map.bounds`` ``[x, y, width, height]`` as the playable ``[min, max]`` per axis."""
    if not (isinstance(bounds, list) and len(bounds) == 4 and all(map(_is_int, bounds))):
        return None
    x, y, width, height = bounds
    if width < 1 or height < 1:
        return None
    return {"x": [x, x + width - 1], "y": [y, y + height - 1]}


# --- Order fields, the answer schema and validation ------------------------------


def order_fields(config: dict[str, Any] | None) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """cmd -> (required, optional) fields: the config's ``order_schema.orders`` when present, else ORDER_FIELDS."""
    orders = _dict(_dict(_dict(config).get("order_schema")).get("orders"))
    if not orders:
        return dict(ORDER_FIELDS)
    fields: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for cmd, entry in orders.items():
        names = frozenset(_dict(_dict(entry).get("fields")))
        if not names and cmd in ORDER_FIELDS:
            fields[cmd] = ORDER_FIELDS[cmd]
            continue
        required = ORDER_FIELDS.get(cmd, (frozenset(), frozenset()))[0] & names
        fields[str(cmd)] = (required, names - required)
    return fields


def answer_schema(fields: dict[str, tuple[frozenset[str], frozenset[str]]]) -> dict:
    """The model's strict answer: ``orders`` (one object shape per command;
    optional fields are present but may be null) and ``reasoning_summary``."""
    shapes = []
    for cmd, (required, optional) in fields.items():
        props: dict[str, Any] = {"cmd": {"type": "string", "enum": [cmd]}}
        for name in _field_order(required):
            props[name] = FIELD_SCHEMAS.get(name, {"type": "string"})
        for name in _field_order(optional):
            if name in FIELD_SCHEMAS:
                props[name] = {"anyOf": [FIELD_SCHEMAS[name], {"type": "null"}]}
        shapes.append({"type": "object", "properties": props, "required": list(props), "additionalProperties": False})
    return object_schema({"orders": {"type": "array", "items": {"anyOf": shapes}}})


def _field_ok(name: str, value: Any, limits: dict[str, int]) -> bool:
    if name == "units":
        return (
            isinstance(value, list)
            and 1 <= len(value) <= limits["max_units_per_order"]
            and all(_is_int(u) and u >= 0 for u in value)
            and len(set(value)) == len(value)
        )
    if name in ("to", "at"):
        return _is_cell(value)
    if name in ("target", "building"):
        return _is_int(value) and value >= 1
    if name == "item":
        return isinstance(value, str) and 1 <= len(value.strip()) <= MAX_ITEM_NAME
    if name == "count":
        return _is_int(value) and 1 <= value <= limits["max_train_count"]
    if name in ("queued", "on"):
        return isinstance(value, bool)
    if name == "stance":
        return isinstance(value, str) and bool(value.strip())
    return True  # a field this guard does not know: the server decides


def order_ok(order: Any, fields: dict[str, tuple[frozenset[str], frozenset[str]]], limits: dict[str, int]) -> bool:
    """Cheap shape check of one order (the server rejects unknown fields too)."""
    if not isinstance(order, dict):
        return False
    cmd = order.get("cmd")
    if not isinstance(cmd, str) or cmd not in fields:
        return False
    required, optional = fields[cmd]
    keys = set(order) - {"cmd"}
    if not required <= keys or keys - required - optional:
        return False
    if cmd == "attack" and ("target" in order) == ("to" in order):
        return False  # exactly one of target / to
    return all(_field_ok(k, order[k], limits) for k in keys)


def validate_orders(
    raw: list[Any], fields: dict[str, tuple[frozenset[str], frozenset[str]]], limits: dict[str, int]
) -> tuple[list[dict[str, Any]], int]:
    """(the well-formed orders, cut to max_orders_per_batch; how many were dropped).
    Null fields (the schema's way of leaving an optional field out) are removed first."""
    cleaned = [{k: v for k, v in o.items() if v is not None} if isinstance(o, dict) else o for o in raw]
    batch = [o for o in cleaned if order_ok(o, fields, limits)][: limits["max_orders_per_batch"]]
    return batch, len(raw) - len(batch)


# --- Prompts ----------------------------------------------------------------------

_FIELD_ORDER = ("units", "building", "item", "target", "to", "at", "count", "stance")


def _field_order(names: frozenset[str]) -> list[str]:
    known = [f for f in _FIELD_ORDER if f in names]
    return known + sorted(names - set(known))


def _order_lines(config: dict[str, Any], fields: dict[str, tuple[frozenset[str], frozenset[str]]]) -> list[str]:
    schema_orders = _dict(_dict(config.get("order_schema")).get("orders"))
    lines = []
    for cmd, (required, optional) in fields.items():
        names = [f for f in _field_order(required | optional) if f in required]
        names += [f"{f}?" for f in _field_order(optional)]
        line = f"- {cmd}: {', '.join(names)}"
        if cmd == "attack":
            line += " (exactly one of target / to)"
        notes = _dict(schema_orders.get(cmd)).get("notes")
        if isinstance(notes, str):
            line += f" - {_text(notes, '', 160)}"
        lines.append(line)
    return lines


def _stances(config: dict[str, Any]) -> str:
    orders = _dict(_dict(config.get("order_schema")).get("orders"))
    stance = _dict(_dict(orders.get("set_stance")).get("fields")).get("stance")
    return _text(stance, " | ".join(DEFAULT_STANCES), 120)


def _map_terms(config: dict[str, Any]) -> str:
    terms = _text(_dict(config.get("rules_glossary")).get("maps"), FALLBACK_MAP_TERMS)
    return (
        f"Map: {terms.rstrip('.')}. map.bounds is [x, y, width, height]: playable "
        "cells are x..x+width-1, y..y+height-1 (map.playable in each state gives "
        "them as [min, max] per axis)."
    )


def build_system_prompt(config: dict[str, Any] | None) -> str:
    """The standing instructions: goal, real time, order formats, build basics and
    the answer format. Rule text comes from the game config when the runner fetched it."""
    config = _dict(config)
    results = _dict(config.get("results_policy"))
    realtime = _dict(config.get("realtime"))
    max_minutes = _dict(_dict(config.get("timeouts")).get("max_minutes")).get("max")
    limits = _limits({"limits": _dict(_dict(config.get("order_schema")).get("limits"))})
    fields = order_fields(config)
    buildings = _dict(_dict(config.get("rules")).get("buildings"))
    limit_line = (
        f"The match has a time limit (at most {max_minutes} minutes; this match's remaining_seconds is in each state). "
        if max_minutes
        else "The match has a time limit (remaining_seconds is in each state). "
    )
    war_factory = "a war factory (weap) for vehicles, " if not buildings or "weap" in buildings else ""
    examples = "\n".join(_dumps(e) for e in ORDER_EXAMPLES)
    sections = [
        "You command one side of a 1v1 Command & Conquer: Red Alert match (OpenRA engine) against "
        "another AI agent. Each turn you get your current view of the game as JSON and answer with "
        "a batch of orders.",
        "GOAL\n"
        f"- {_text(results.get('win'), FALLBACK_WIN)}\n"
        f"- {limit_line}At the limit: "
        f"{_text(results.get('time_limit_and_idle_tiebreak'), FALLBACK_TIEBREAK)}. "
        "kills_cost is the value you destroyed, deaths_cost the value you lost, assets_value what "
        "you still own (economy/military in the state).",
        "REAL TIME\n"
        f"- {_text(realtime.get('world_never_waits'), FALLBACK_REALTIME)}\n"
        "- Your answer takes time, and the game runs on meanwhile: a slow answer acts on a stale "
        "view. Prefer short, decisive batches; you will be asked again as soon as your orders are sent.",
        "ORDERS (fields with ? are optional; any other field makes the order invalid)\n"
        + "\n".join(_order_lines(config, fields))
        + "\n"
        f"units: 1-{limits['max_units_per_order']} of your own unit ids. to/at: a [x, y] map cell "
        f"inside map.bounds. {_map_terms(config)} target: an actor id (attack: a visible enemy; "
        "guard/enter_transport: your own). building: your building id. item: a lowercase name from "
        f"available_production. count: 1-{limits['max_train_count']}. queued: true/false. stance: "
        f"{_stances(config)}.\n"
        f"At most {limits['max_orders_per_batch']} orders per answer. Each order is checked on its "
        "own (a bad one is rejected, the rest still run); verdicts arrive in last_orders and "
        "recent_order_problems of later states.\n"
        f"Examples:\n{examples}",
        "BUILD BASICS\n"
        f"- First deploy your MCV ({MCV}) into a construction yard ({CONSTRUCTION_YARD}); nothing "
        "can be built before that. Deploy it once: the yard is a building with a new id, and "
        "buildings cannot be deployed.\n"
        f"- Then a power plant ({POWER_PLANT}), barracks ({' or '.join(BARRACKS)}, by faction), a "
        f"refinery ({REFINERY}; it comes with a harvester), {war_factory}more power plants whenever "
        "power_drained exceeds power_provided (low power slows everything).\n"
        "- A finished building waits in its queue (ready: true) until you place it: place without "
        "at puts it next to your base.\n"
        f"- Train units (e.g. {INFANTRY} rifle infantry from barracks), keep harvesters working, and "
        "attack-move groups at the enemy base; defend when under_attack events arrive. Costs, power "
        "and build times of what you can build now are in available_production.",
        *([f"STRATEGY PLAYBOOK (follow it over the build basics above)\n{guide}"] if (guide := playbook("red_alert")) else []),
        "ANSWER\n"
        "orders: the batch to send now, each order in one of the shapes above (an optional field "
        "you don't use is null). An empty list means nothing to do right now.\n"
        f"reasoning_summary: your plan in one or two sentences, at most {REASONING_MAX_CHARS} "
        "characters; spectators see it after the match.",
    ]
    return "\n\n".join(sections)


def _item_facts(name: str, rules: dict[str, Any]) -> dict[str, Any]:
    """An available_production entry with cost/power/build time from the rules."""
    key = name.lower()
    info = _dict(_dict(rules.get("buildings")).get(key)) or _dict(_dict(rules.get("units")).get(key))
    entry: dict[str, Any] = {"item": key}
    if info.get("queue"):
        entry["queue"] = info["queue"]
    if info.get("cost") is not None:
        entry["cost"] = info["cost"]
    if info.get("power"):
        entry["power"] = info["power"]
    if info.get("build_time_s"):
        entry["build_s"] = info["build_time_s"]
    return entry


def _actor(actor: dict[str, Any], *, idle: bool = False) -> dict[str, Any]:
    cell = _cell(actor)
    entry: dict[str, Any] = {"id": actor.get("id"), "type": actor.get("type"), "cell": list(cell) if cell else None, "hp": actor.get("hp")}
    if idle:
        entry["idle"] = bool(actor.get("idle"))
    return entry


def _last_orders(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    results = [
        {k: r.get(k) for k in ("index", "cmd", "status", "reason")} for r in value.get("results") or [] if isinstance(r, dict)
    ]
    return {"batch": value.get("batch"), "pending": bool(value.get("pending")), "results": results[:MAX_ORDER_RESULTS_SHOWN]}


def _problems(value: Any) -> list[dict[str, Any]]:
    rows = [
        {
            "cmd": p.get("cmd"),
            "status": p.get("status"),
            "reason": p.get("reason"),
            "detail": _text(p.get("detail"), "", DETAIL_MAX_CHARS) or None,
        }
        for p in value or []
        if isinstance(p, dict)
    ]
    return rows[-MAX_PROBLEMS_SHOWN:]


def _map_section(state: dict[str, Any], obs: dict[str, Any], map_info: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**_dict(state.get("map")), **(map_info or {})}
    base = _cell(obs.get("base_center"))
    spawns = _spawns(merged)
    guess = choose_enemy_spawn(spawns, base)
    return {
        "id": merged.get("id"),
        "width": merged.get("width"),
        "height": merged.get("height"),
        "bounds": merged.get("bounds"),
        "playable": playable_cells(merged.get("bounds")),
        "spawns": [list(s) for s in spawns],
        "base_center": list(base) if base else None,
        "enemy_spawn_guess": list(guess) if guess else None,
    }


def _capped(items: list[Any], cap: int, key: str, omitted: dict[str, int]) -> list[Any]:
    if len(items) > cap:
        omitted[key] = omitted.get(key, 0) + len(items) - cap
    return items[:cap]


def _sections(
    state: dict[str, Any],
    map_info: dict[str, Any] | None,
    config: dict[str, Any] | None,
    last_reasoning: str | None,
    caps: dict[str, int],
    refused: list[str] | None = None,
    refused_in_row: int = 0,
    repeats: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """The user prompt's sections in order, and how many list entries were cut."""
    obs = _dict(state.get("observation"))
    rules = _dict(_dict(config).get("rules"))
    time_info = _dict(state.get("time"))
    omitted: dict[str, int] = {}
    enemy_buildings = [{**_actor(e), "kind": "building"} for e in _list(obs, "visible_enemy_buildings")]
    enemy_units = [{**_actor(e), "kind": "unit"} for e in _list(obs, "visible_enemies")]
    available = [_item_facts(str(x), rules) for x in obs.get("available_production") or []]
    production = [
        {k: p.get(k) for k in ("queue", "item", "progress", "remaining_s", "ready", "paused") if k != "paused" or p.get("paused")}
        for p in _list(obs, "production")
    ]
    events = [e for e in obs.get("events") or [] if isinstance(e, dict)]
    sections: dict[str, Any] = {
        "time": {
            "tick": _obs_tick(state),
            "game_seconds": time_info.get("game_seconds"),
            "remaining_seconds": time_info.get("remaining_seconds"),
            "max_minutes": time_info.get("max_minutes"),
        },
        "you": {"seat": state.get("your_seat"), "faction": obs.get("faction"), "enemy_faction": obs.get("enemy_faction")},
        "economy": _dict(obs.get("economy")),
        "military": _dict(obs.get("military")),
        "units": _capped([_actor(u, idle=True) for u in _list(obs, "units")], caps["units"], "units", omitted),
        "buildings": _capped([_actor(b) for b in _list(obs, "buildings")], caps["buildings"], "buildings", omitted),
        "production": _capped(production, caps["production"], "production", omitted),
        "available_production": _capped(available, caps["available_production"], "available_production", omitted),
        "enemies": _capped(enemy_buildings + enemy_units, caps["enemies"], "enemies", omitted),
        "map": _map_section(state, obs, map_info),
        "last_orders": _last_orders(state.get("last_orders")),
        "recent_order_problems": _problems(state.get("recent_order_problems")),
        "events": events[-MAX_EVENTS_SHOWN:],
        "your_last_reasoning": last_reasoning,
        "limits": _limits(state),
    }
    # Right after the clock, and never dropped for space.
    warnings: dict[str, Any] = {}
    if refused:
        warnings["your_refused_batches"] = {"in_a_row": refused_in_row, "reasons_newest_last": list(refused), "note": REFUSED_NOTE}
    if repeats:
        warnings["orders_refused_repeatedly"] = {"orders": list(repeats), "note": REPEATS_NOTE}
    attack_now = _dict(state.get("legal_actions")).get("attack_now")
    if isinstance(attack_now, dict) and attack_now:
        warnings["suggested_attack"] = {"order": attack_now, "note": SUGGESTED_ATTACK_NOTE}
    queued = over_queued(state)
    if queued:
        warnings["already_queued_many_times"] = {"items": queued, "note": OVERQUEUE_NOTE}
    if warnings:
        sections = {"time": sections.pop("time"), **warnings, **sections}
    return sections, omitted


def _render(sections: dict[str, Any], note: dict[str, Any]) -> str:
    body = {"note": note, **sections} if note else sections
    return f"Your current view of the game (JSON):\n{_dumps(body)}\nAnswer with orders and reasoning_summary."


def build_user_prompt(
    state: dict[str, Any],
    map_info: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    last_reasoning: str | None = None,
    refused: list[str] | None = None,
    refused_in_row: int = 0,
    repeats: list[dict[str, Any]] | None = None,
    max_chars: int = USER_PROMPT_MAX_CHARS,
) -> str:
    """A compact JSON view of ``state`` (GameAPI's raw payload) for the model, at
    most ``max_chars``. Lists are capped and the cut is noted; when the text is
    still too long, low-priority sections are dropped first, then lists shortened."""
    caps = {
        "units": MAX_UNITS_SHOWN,
        "buildings": MAX_BUILDINGS_SHOWN,
        "enemies": MAX_ENEMIES_SHOWN,
        "production": MAX_PRODUCTION_SHOWN,
        "available_production": MAX_AVAILABLE_SHOWN,
    }
    dropped: list[str] = []
    while True:
        sections, omitted = _sections(state, map_info, config, last_reasoning, caps, refused, refused_in_row, repeats)
        for key in dropped:
            sections.pop(key, None)
        note: dict[str, Any] = {}
        if omitted:
            note["entries_not_shown"] = omitted
        if dropped:
            note["sections_left_out_to_save_space"] = list(dropped)
        text = _render(sections, note)
        if len(text) <= max_chars:
            return text
        droppable = [k for k in LOW_PRIORITY_SECTIONS if k not in dropped]
        if droppable:
            dropped.append(droppable[0])
            continue
        shrinkable = [k for k in SHRINKABLE_LISTS if caps[k] > MIN_LIST_SHOWN]
        if shrinkable:
            caps[shrinkable[0]] = max(MIN_LIST_SHOWN, caps[shrinkable[0]] // 2)
            continue
        minimal = {k: sections[k] for k in ("time", "you", "economy") if k in sections}
        text = _render(minimal, {"sections_left_out_to_save_space": "all but time, you, economy"})
        return text[:max_chars]


# --- Repeated refusals --------------------------------------------------------------


def order_key(order: dict[str, Any]) -> str:
    """An order's identity for counting refusals: its content without ``queued``, unit ids sorted."""
    body = {k: v for k, v in order.items() if k != "queued"}
    if isinstance(body.get("units"), list):
        body["units"] = sorted(body["units"], key=str)
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


def refusal_rows(detail: Any) -> list[tuple[int, str, str, str | None]]:
    """(index, cmd, reason, detail) for each order in a whole-batch refusal's detail text."""
    if not isinstance(detail, str):
        return []
    return [(int(m.group(1)), m.group(2), m.group(3), m.group(4)) for m in _REFUSAL_ROW.finditer(detail)]


def reason_still_holds(entry: dict[str, Any], state: dict[str, Any]) -> bool:
    """Whether a refused order would most likely be refused again for the same
    reason, judged from the current state (True when it cannot be judged)."""
    obs = _dict(state.get("observation"))
    order, reason = entry["order"], entry["reason"]
    item = str(order.get("item") or "").lower()
    if reason == "not_ready" and order.get("cmd") == "place":
        return not any(str(p.get("item") or "").lower() == item and p.get("ready") for p in _list(obs, "production"))
    if reason == "not_available" and order.get("cmd") in ("build", "train"):
        return item not in {str(x).lower() for x in obs.get("available_production") or []}
    if reason in ("is_building", "not_owned", "wreck") and order.get("units"):
        own = {u.get("id") for u in _list(obs, "units") if not u.get("wreck")}
        return not set(order["units"]) <= own
    return True


def over_queued(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Buildings queued OVERQUEUE_WARN_AT times or more and not ready yet."""
    counts: dict[str, int] = {}
    for p in _list(_dict(state.get("observation")), "production"):
        if str(p.get("queue")).lower() in ("building", "defense") and not p.get("ready"):
            item = str(p.get("item") or "").lower()
            counts[item] = counts.get(item, 0) + 1
    return [{"item": i, "queued": n} for i, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= OVERQUEUE_WARN_AT]


def drop_requeued_buildings(orders: list[dict[str, Any]], state: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """``orders`` without a ``build`` of a building or defense that is already in production
    (queued, in progress or ready to place), or built twice in one batch. Test games: the
    model queued two refineries and two barracks at once again and again, tying up all the
    cash. A second refinery is still fine once the first has been placed."""
    busy = {str(p.get("item") or "").lower() for p in _list(_dict(state.get("observation")), "production")
            if str(p.get("queue")).lower() in ("building", "defense")}
    kept, dropped = [], 0
    for order in orders:
        item = str(order.get("item") or "").lower()
        if order.get("cmd") == "build" and item and item in busy:
            dropped += 1
            continue
        if order.get("cmd") == "build" and item:
            busy.add(item)
        kept.append(order)
    return kept, dropped


# --- The player ---------------------------------------------------------------------


class RedAlertPlayer:
    """One Red Alert match's decisions. ``choose_action`` returns
    ``WithReasoning({"type": "orders", "orders": [...]}, reasoning)`` or
    ``WAIT``; ``on_action_result`` learns from each ``play_action`` answer.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        log: Callable[[str], None] = print,
        max_calls: int = MAX_CALLS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider = provider
        self._log = log
        self.max_calls = max_calls
        self._clock = clock
        self.calls = 0
        self.ok = 0
        self.fallbacks: dict[str, int] = {}
        self.orders_proposed = 0
        self.orders_dropped = 0
        self.orders_blocked = 0
        self._config: dict[str, Any] | None = None
        self._fields = order_fields(None)
        self._schema = answer_schema(self._fields)
        self._system: str | None = None
        self._map_id: str | None = None
        self._map_info: dict[str, Any] | None = None
        self._last_reasoning: str | None = None
        # Batches refused as a whole, newest last, and how many in a row;
        # cleared by the next accepted batch.
        self._refused: list[str] = []
        self._refused_in_row = 0
        # The orders last sent (the batch play_action answers next), and every
        # order refused recently: order_key -> {order, reason, detail, count, last}.
        self._last_sent: list[dict[str, Any]] = []
        self._refusals: dict[str, dict[str, Any]] = {}
        self._errors_in_row = 0
        self._backoff_until = 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "ok": self.ok,
            "fallbacks": dict(self.fallbacks),
            "orders_proposed": self.orders_proposed,
            "orders_dropped": self.orders_dropped,
            "orders_blocked": self.orders_blocked,
        }

    # -- decisions -------------------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        raw = state.raw
        if not isinstance(raw.get("observation"), dict):
            return WAIT  # nothing to look at yet
        self._use_config(context.game_config)
        if self.calls >= self.max_calls:
            return self._fallback(FALLBACK_CAP)
        if self._clock() < self._backoff_until:
            return self._fallback(FALLBACK_BACKOFF)
        user = build_user_prompt(
            raw,
            self._map(raw),
            self._config,
            # Its own last plan is left out while its batches are being refused:
            # fed back, it tends to lock the model into the same refused move.
            last_reasoning=None if self._refused else self._last_reasoning,
            refused=self._refused,
            refused_in_row=self._refused_in_row,
            repeats=self._repeats(raw),
        )
        self.calls += 1
        messages = [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": user}]
        try:
            answer = self._provider.complete_structured(messages, "red_alert_orders", self._schema)
        except ProviderError as exc:
            self._errors_in_row += 1
            pause = backoff_seconds(self._errors_in_row)
            self._backoff_until = self._clock() + pause
            return self._fallback(FALLBACK_ERROR, f"{exc}; no model call for {pause:g} s")
        self._errors_in_row = 0
        return self._use_answer(raw, answer)

    def _use_answer(self, raw: dict[str, Any], answer: dict):
        proposed = answer.get("orders")
        if not isinstance(proposed, list):
            return self._fallback(FALLBACK_BAD_ANSWER, "orders is not a list")
        orders, dropped = validate_orders(proposed, self._fields, _limits(raw))
        self.orders_proposed += len(proposed)
        self.orders_dropped += dropped
        if proposed and not orders:
            return self._fallback(FALLBACK_BAD_ANSWER, f"all {len(proposed)} orders malformed")
        self.ok += 1
        # Orders refused BLOCK_AFTER times for a reason that still holds are not
        # sent (the next prompt marks them blocked); the rest go out.
        orders = self._drop_blocked(orders, raw)
        orders, requeued = drop_requeued_buildings(orders, raw)
        if requeued:
            self._log(f"[llm] red_alert: dropped {requeued} build order(s) for buildings already in production")
        reasoning = trim_reasoning(answer.get("reasoning_summary"))
        self._last_reasoning = reasoning
        if not orders:
            return WAIT
        self._last_sent = orders
        summary = reasoning or f"{len(orders)} order(s); the model gave no reasoning"
        self._log(f"[llm] red_alert: {len(orders)} order(s): {summary}")
        return WithReasoning({"type": "orders", "orders": orders}, summary)

    def _fallback(self, reason: str, detail: str | None = None):
        count = self.fallbacks.get(reason, 0) + 1
        self.fallbacks[reason] = count
        # cap and backoff repeat on every decision while they last: log the first.
        if reason not in (FALLBACK_CAP, FALLBACK_BACKOFF) or count == 1:
            self._log(f"[llm] red_alert: FALLBACK ({reason}), nothing sent" + (f": {detail[:200]}" if detail else ""))
        return WAIT

    # -- play_action answers -------------------------------------------------------------

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        """A batch refused as a whole is remembered with its reasons (the server
        records it nowhere else); each refused order is counted; an accepted
        batch clears the refused list."""
        sent = self._last_sent
        if result.get("error") == "INVALID_ACTION":
            detail = _text(result.get("detail"), "no order was valid", REFUSED_DETAIL_MAX_CHARS)
            self._refused = [*self._refused, detail][-MAX_REFUSED_SHOWN:]
            self._refused_in_row += 1
            for index, cmd, reason, why in refusal_rows(result.get("detail")):
                if index < len(sent) and sent[index].get("cmd") == cmd:
                    self._record_refusal(sent[index], reason, why)
            return
        if "error" in result:
            return  # a race (stale view, batch still in flight): nothing to learn
        if result.get("accepted"):
            self._refused, self._refused_in_row = [], 0
        for row in result.get("results") or []:
            index = row.get("index") if isinstance(row, dict) else None
            if not _is_int(index) or not 0 <= index < len(sent):
                continue
            if row.get("status") == "rejected":
                self._record_refusal(sent[index], row.get("reason"), row.get("detail"))
            elif row.get("status") in ("ok", "partial", "pending"):
                self._refusals.pop(order_key(sent[index]), None)

    def _record_refusal(self, order: dict[str, Any], reason: Any, detail: Any) -> None:
        key = order_key(order)
        reason = str(reason or "rejected")
        entry = self._refusals.get(key)
        count = entry["count"] + 1 if entry and entry["reason"] == reason else 1
        self._refusals[key] = {
            "order": order,
            "reason": reason,
            "detail": _text(detail, "", DETAIL_MAX_CHARS) or None,
            "count": count,
            "last": self._clock(),
        }

    def _blocked(self, entry: dict[str, Any], raw: dict[str, Any]) -> bool:
        return (
            entry["count"] >= BLOCK_AFTER
            and self._clock() - entry["last"] < BLOCK_SECONDS
            and reason_still_holds(entry, raw)
        )

    def _repeats(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """The prompt's orders_refused_repeatedly rows (forgets old refusals)."""
        now = self._clock()
        self._refusals = {k: e for k, e in self._refusals.items() if now - e["last"] < REFUSAL_FORGET_SECONDS}
        rows = [
            {"order": e["order"], "refused": e["count"], "reason": e["reason"], "detail": e["detail"], "blocked": self._blocked(e, raw)}
            for e in self._refusals.values()
            if e["count"] >= REPEAT_WARN_AFTER
        ]
        rows.sort(key=lambda r: -r["refused"])
        return rows[:MAX_REPEATS_SHOWN]

    def _drop_blocked(self, orders: list[dict[str, Any]], raw: dict[str, Any]) -> list[dict[str, Any]]:
        kept = []
        for order in orders:
            entry = self._refusals.get(order_key(order))
            if entry is not None and self._blocked(entry, raw):
                self.orders_blocked += 1
                continue
            kept.append(order)
        return kept

    # -- setup ------------------------------------------------------------------------------

    def _use_config(self, config: dict | None) -> None:
        if config is self._config or not isinstance(config, dict):
            return
        self._config = config
        self._fields = order_fields(config)
        self._schema = answer_schema(self._fields)
        self._system = None
        self._map_id = None

    def _system_prompt(self) -> str:
        if self._system is None:
            self._system = build_system_prompt(self._config)
        return self._system

    def _map(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        map_id = _dict(raw.get("map")).get("id")
        if map_id != self._map_id:
            self._map_id = map_id
            self._map_info = find_map_info(self._config, map_id)
        return self._map_info
