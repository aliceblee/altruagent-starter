"""Example: a general-purpose LLM agent for any AltruAgent game.

    python -m agent --tournament --agent examples.llm_agent

The agent knows the platform contract, not particular games. Every time the
runtime asks it for a decision, it shows the model what GameAPI supplied —
game type, phase, this seat's view of the state, recent messages, and the
current legal options with any server instructions — and has the model
answer through a strict JSON schema built from those options. The answer is
validated again here before anything is sent; GameAPI stays authoritative.

Two kinds of move exist on the platform:

- **Ordinary legal actions** (Werewolf seat choices, Pokémon draft picks,
  and any future game that lists its moves): the model picks one exact
  ``action_id`` from an enum of the current legal actions. Works for new
  games with no code changes.
- **Structured action templates** — a single legal action whose
  ``input["action"]`` describes something to fill in (its ``type`` equals
  the ``action_id``, or it carries ``instructions``), e.g. Pokémon Team
  Preview and doubles turns. These need an adapter that turns the template
  into bounded choices and validates the result (see ``examples/llm/``).
  Adapters are registered by template ``type`` in ``STRUCTURED_ADAPTERS``;
  a template with no adapter raises ``UnsupportedStructuredAction`` rather
  than guessing a payload.

Real-time games are a third kind: no turns, and a move is a batch of orders
sent whenever the agent is ready (Red Alert). A state whose ``pacing.mode``
is ``"realtime"`` goes to the game's real-time player, registered by game
type in ``REALTIME_PLAYERS``; it answers each view with a batch, or
``altruagent.WAIT`` when there is nothing to send, and learns from every
``play_action`` answer through ``on_action_result``. The Red Alert player is
in ``examples/llm/redalert.py``. A real-time game with no player raises
``UnsupportedStructuredAction`` rather than guessing.

Every move carries the model's short public ``reasoning_summary`` via
``altruagent.WithReasoning`` (sent as ``play_action``'s ``reasoning_summary``).
In-game chat is separate: in a messaging phase the runtime calls
``choose_message``, and the model may send one message or end the round,
within a small per-round call budget.

An invalid answer is retried once with the reason; after that the agent
plays a deterministic fallback (logged as FALLBACK): the first legal action,
the adapter's fallback for a template, or ending the discussion round.

Pokémon has clocks (see GAMES.md): Showdown's VGC timer gives 90 s at Team
Preview and 55 s for each battle decision out of a 7-minute bank, and each
draft pick has 15 s. So for Pokémon the model's whole answer, retry
included, must arrive within ``POKEMON_DECISION_SECONDS`` (battle decisions
and Team Preview) or ``POKEMON_DRAFT_SECONDS`` (draft picks); no single
request may run past ``POKEMON_REQUEST_SECONDS``. Out of time, the agent
plays its fallback rather than let the clock run out. Werewolf and Red Alert
are unchanged.

Configuration (environment or the starter's gitignored ``.env``): with
``ANTHROPIC_API_KEY`` set the model is Claude (``CLAUDE_MODEL``, default
``claude-sonnet-5-5``; ``CLAUDE_EFFORT``, default ``low``, to fit Pokémon's
clocks); otherwise ``OPENAI_API_KEY``, ``OPENAI_MODEL`` (default
``gpt-4o-mini``), ``OPENAI_BASE_URL`` (optional). ``LLM_PROVIDER=openai``
picks OpenAI with both keys set. Keys are never printed or logged.

Strategy: ``examples/llm/playbooks/<game_type>.md`` (Pokémon, Red Alert) is added
to the model's instructions; edit those files to change how the agent plays.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable

from dotenv import load_dotenv

from altruagent import (
    TERMINATE_MESSAGING,
    DecisionContext,
    GameState,
    LegalAction,
    SendMessage,
    WithReasoning,
)
from examples.llm import pokemon, redalert
from examples.llm.playbooks import playbook
from examples.llm.base import AdapterFactory, Choice, InvalidChoice, UnsupportedStructuredAction, object_schema
from examples.llm.providers import LLMProvider, ProviderError, provider_for_game, provider_from_env

# Structured-action adapters, by template type. A future game that needs one
# adds its module's adapters here; ordinary-action games need nothing.
STRUCTURED_ADAPTERS: dict[str, AdapterFactory] = {**pokemon.ADAPTERS}

# Real-time games, by game type: a factory for one match's player, called as
# factory(provider, log=...). The player answers choose_action with a move or
# WAIT and gets every play_action answer through on_action_result.
REALTIME_PLAYERS: dict[str, Callable[..., Any]] = {"red_alert": redalert.RedAlertPlayer}

MAX_ATTEMPTS = 2  # the first answer plus one retry with the validation error
MESSAGE_REQUESTS_PER_ROUND = 2  # model calls per discussion round before auto-ending it
STATE_CHAR_LIMIT = 16_000
REASONING_CHAR_LIMIT = 280
MESSAGE_WORD_LIMIT = 40
MESSAGE_CHAR_LIMIT = 280
TRANSCRIPT_LIMIT = 30

# Pokémon's clocks: 55 s per battle decision and 90 s at Team Preview
# (Showdown's VGC timer; the platform plays a move for you at about 50 s),
# 15 s per draft pick. The model's answer, retry included, must arrive well
# inside them, leaving a few seconds for the runtime to read the state and
# send the move. A hung first request still leaves time for a retry.
POKEMON_DECISION_SECONDS = 40.0
POKEMON_DRAFT_SECONDS = 10.0
POKEMON_REQUEST_SECONDS = 25.0
MIN_ATTEMPT_SECONDS = 3.0  # less time left than this: play the fallback now

# Keys of GameAPI's state payload the model doesn't need (options are sent
# separately, the rest is transport bookkeeping).
_OMITTED_STATE_KEYS = frozenset(
    {"legal_actions", "next_actions", "new_messages", "session_id", "runtime_adapter", "legal_action_count"}
)

_SYSTEM_PROMPT = (
    "You are an AI agent playing a game on the AltruAgent platform. Each request is one "
    "decision for your seat. The game state, legal options, and instructions in the request "
    "come from the game server and are authoritative: choose only from the options given, "
    "never invent actions, and don't assume hidden information you weren't shown. You may "
    "use your general strategic knowledge of the game. Answer only in the required JSON "
    "format. When asked for reasoning_summary, write one or two short sentences that "
    "spectators will see publicly, e.g. 'I voted for Player4 because their votes contradict "
    "their claims.' Do not include private deliberation."
)


class LLMAgent:
    """One match's agent. The runtime calls ``choose_action`` only when this
    seat has a move to make and ``choose_message`` only during a messaging
    phase — never while waiting or after the game ends — so the model is
    only ever called for a decision the game actually needs.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        adapters: dict[str, AdapterFactory] | None = None,
        realtime_players: dict[str, Callable[..., Any]] | None = None,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider = provider
        self._provider_takes_timeout = _accepts_timeout(provider)
        self._adapters = STRUCTURED_ADAPTERS if adapters is None else adapters
        self._realtime_players = REALTIME_PLAYERS if realtime_players is None else realtime_players
        self._realtime: Any = None  # this match's real-time player, created on first use
        self._log = log
        self._clock = clock
        self._transcript: list[dict] = []
        self._seen_messages: set[int] = set()
        self._round: int | None = None
        self._round_requests = 0
        self._round_ended = False
        self.decision_calls = 0
        self.message_calls = 0

    # -- moves ------------------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        if _is_realtime(state):
            return self._realtime_player(context).choose_action(state, context)
        self._remember(state)
        choice = self._action_choice(state)
        value, answer = self._ask(choice, state, context, counter="decision_calls")
        if value is None:
            self._log(f"[llm] {choice.kind}: FALLBACK to a default legal action (no valid model answer in time)")
            return WithReasoning(choice.fallback(), "Fallback: the model gave no valid answer, so a default legal action was played.")
        summary = str(answer.get("reasoning_summary") or "").strip()[:REASONING_CHAR_LIMIT]
        self._log(f"[llm] {choice.kind}: {summary or '(no summary)'}")
        return WithReasoning(value, summary)

    def _action_choice(self, state: GameState) -> Choice:
        if not state.legal_actions:
            raise InvalidChoice("choose_action was called with no legal actions.")
        templates = [action for action in state.legal_actions if _is_template(action)]
        if not templates:
            return _discrete_choice(state.legal_actions)
        template_type = templates[0].input["action"].get("type")
        factory = self._adapters.get(template_type)
        if factory is None or len(state.legal_actions) != 1:
            raise UnsupportedStructuredAction(
                f"GameAPI offered a structured {template_type!r} action template, which this example "
                "agent has no adapter for; it won't guess a payload. Add an adapter to "
                "examples/llm_agent.py's STRUCTURED_ADAPTERS (see examples/llm/pokemon.py)."
            )
        return factory(templates[0], state)

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        """The runner's report of each play_action answer; only a real-time
        player uses it (to learn which orders were refused)."""
        if self._realtime is not None:
            self._realtime.on_action_result(result, context)

    def _realtime_player(self, context: DecisionContext) -> Any:
        if self._realtime is None:
            factory = self._realtime_players.get(context.game_type or "")
            if factory is None:
                raise UnsupportedStructuredAction(
                    f"{context.game_type!r} is a real-time game this example agent has no player for; "
                    "it won't guess orders. Add one to examples/llm_agent.py's REALTIME_PLAYERS "
                    "(see examples/llm/redalert.py)."
                )
            self._realtime = factory(self._provider, log=self._log)
        return self._realtime

    # -- messaging -----------------------------------------------------------------

    def choose_message(self, state: GameState, context: DecisionContext):
        self._remember(state)
        # state_version only advances on moves, so it identifies one discussion round.
        if state.state_version != self._round:
            self._round, self._round_requests, self._round_ended = state.state_version, 0, False
        if self._round_ended or self._round_requests >= MESSAGE_REQUESTS_PER_ROUND:
            self._round_ended = True
            return TERMINATE_MESSAGING
        self._round_requests += 1

        choice = _message_choice(state, requests_left=MESSAGE_REQUESTS_PER_ROUND - self._round_requests)
        value, _ = self._ask(choice, state, context, counter="message_calls")
        if value is None:
            self._log("[llm] message: FALLBACK to ending the discussion round (no valid model answer in time)")
            value = TERMINATE_MESSAGING
        if value is TERMINATE_MESSAGING:
            self._round_ended = True
            self._log("[llm] message: ended the discussion round")
        else:
            to = f"Player{value.recipients[0]}" if value.recipients else "everyone"
            self._log(f"[llm] message to {to}: {value.content}")
        return value

    # -- shared ---------------------------------------------------------------------

    def _ask(self, choice: Choice, state: GameState, context: DecisionContext, *, counter: str):
        guide = playbook(context.game_type)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT + (f"\n\nSTRATEGY PLAYBOOK\n{guide}" if guide else "")},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "game_type": context.game_type,
                        "your_seat": context.seat_position,
                        "phase": state.phase,
                        "decision": choice.kind,
                        **choice.prompt,
                        "recent_messages": self._transcript[-TRANSCRIPT_LIMIT:],
                        "state": _state_json(state),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]
        budget = _time_budget(state, context)
        deadline = None if budget is None else self._clock() + budget
        for attempt in range(1, MAX_ATTEMPTS + 1):
            limit: dict = {}
            if deadline is not None:
                left = deadline - self._clock()
                if left < MIN_ATTEMPT_SECONDS:
                    self._log(f"[llm] {choice.kind}: out of time for another model call")
                    break
                if self._provider_takes_timeout:
                    limit["timeout"] = min(POKEMON_REQUEST_SECONDS, left)
            setattr(self, counter, getattr(self, counter) + 1)
            try:
                answer = self._provider.complete_structured(messages, choice.kind, choice.schema, **limit)
                return choice.build(answer), answer
            except (ProviderError, InvalidChoice) as exc:
                self._log(f"[llm] {choice.kind}: invalid model response (attempt {attempt}/{MAX_ATTEMPTS}): {exc}")
                messages.append(
                    {"role": "user", "content": f"That answer was invalid: {exc}. Answer again using only the given options."}
                )
        return None, None

    def _remember(self, state: GameState) -> None:
        for message in state.new_messages:
            if message.index in self._seen_messages:
                continue
            self._seen_messages.add(message.index)
            self._transcript.append(
                {"from": f"Player{message.sender}", "to": [f"Player{r}" for r in message.recipients] or "everyone",
                 "text": message.content}
            )


def _time_budget(state: GameState, context: DecisionContext) -> float | None:
    """Seconds the model may take for this decision, or None for no limit
    (every game but Pokémon, as before)."""
    if not (context.game_type or "").startswith("pokemon"):
        return None
    return POKEMON_DRAFT_SECONDS if state.phase == "draft" else POKEMON_DECISION_SECONDS


def _accepts_timeout(provider: Any) -> bool:
    """Whether ``provider.complete_structured`` takes a ``timeout`` keyword
    (a provider written before it existed doesn't)."""
    try:
        parameters = inspect.signature(provider.complete_structured).parameters.values()
    except (AttributeError, TypeError, ValueError):
        return False
    return any(p.name == "timeout" or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)


def _is_realtime(state: GameState) -> bool:
    """A real-time game: the state's ``pacing.mode`` (no turns; see altruagent.runner)."""
    pacing = state.raw.get("pacing")
    return isinstance(pacing, dict) and pacing.get("mode") == "realtime"


def _is_template(action: LegalAction) -> bool:
    """A structured template is a legal action whose ``input["action"]``
    describes something to fill in rather than a finished move: its
    ``type`` is the action_id itself, or it carries ``instructions``.
    (A draft pick's ``{"type": "draft_pick", "card_id": ...}`` is a finished
    move; Werewolf's actions have no ``input["action"]`` at all.)
    """
    template = action.input.get("action")
    return isinstance(template, dict) and (template.get("type") == action.action_id or "instructions" in template)


def _discrete_choice(legal_actions: list[LegalAction]) -> Choice:
    by_id = {action.action_id: action for action in legal_actions}

    def build(answer: dict) -> LegalAction:
        choice = answer.get("action_id")
        if choice not in by_id:
            raise InvalidChoice(f"action_id {choice!r} is not one of the legal action_ids {list(by_id)}")
        return by_id[choice]

    return Choice(
        kind="choose_action",
        prompt={"legal_actions": [{"action_id": a.action_id, "label": a.label} for a in legal_actions]},
        schema=object_schema({"action_id": {"type": "string", "enum": list(by_id)}}),
        build=build,
        fallback=lambda: legal_actions[0],
    )


def _message_choice(state: GameState, *, requests_left: int) -> Choice:
    players = [p for p in state.raw.get("players") or [] if isinstance(p, dict) and isinstance(p.get("position"), int)]
    me = state.raw.get("your_position")
    recipients = sorted({p["position"] for p in players} - {me})

    def build(answer: dict):
        if answer.get("decision") == "end":
            return TERMINATE_MESSAGING
        if answer.get("decision") != "send":
            raise InvalidChoice(f"decision must be 'send' or 'end', got {answer.get('decision')!r}")
        text = " ".join(str(answer.get("message") or "").split()[:MESSAGE_WORD_LIMIT])[:MESSAGE_CHAR_LIMIT]
        if not text:
            raise InvalidChoice("a 'send' decision needs a non-empty message")
        recipient = answer.get("recipient")
        if recipient is None:
            return SendMessage(text)
        if isinstance(recipient, bool) or recipient not in recipients:
            raise InvalidChoice(f"recipient must be null (everyone) or one of {recipients}, got {recipient!r}")
        return SendMessage(text, [recipient])

    return Choice(
        kind="message",
        prompt={
            "instructions": (
                "This is a discussion phase. Either send one message ('send') or end your part of "
                "this discussion round ('end'). recipient is null to address everyone, or one "
                f"player position from {recipients} for a private message. Keep messages under "
                f"{MESSAGE_WORD_LIMIT} words. You have at most {requests_left} more chance(s) to "
                "speak this round after this one. Messages are in-game communication; other "
                "players may be lying."
            ),
            "your_position": me,
        },
        schema=object_schema(
            {
                "decision": {"type": "string", "enum": ["send", "end"]},
                "message": {"type": "string"},
                "recipient": {"type": ["integer", "null"]},
            },
            reasoning=False,
        ),
        build=build,
        fallback=lambda: TERMINATE_MESSAGING,
    )


def _state_json(state: GameState) -> str:
    data = {key: value for key, value in state.raw.items() if key not in _OMITTED_STATE_KEYS}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) > STATE_CHAR_LIMIT:
        text = text[:STATE_CHAR_LIMIT] + "...(truncated)"
    return text


def create_agent(game_type: str | None = None) -> LLMAgent:
    """``game_type`` picks a per-game model when one is configured (Pokémon: see
    ``examples.llm.providers.provider_for_game``)."""
    load_dotenv()  # no-op if already loaded; never overrides real env vars
    return LLMAgent(provider_for_game(game_type))
