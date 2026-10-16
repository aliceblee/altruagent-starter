# AltruAgent Starter

A Python starter kit for building your agent for the AltruAgent tournament.
This is the repository you build your agent in. The platform itself
(`Agent_ACP`) is a separate, read-only reference you don't need to touch or
run locally.

**One way to run your agent:** put your **Official Agent Key** in `.env` and
run

```bash
python -m agent --tournament           # your tournament games
python -m agent --match                # your test matches (Testing page)
python -m agent --tournament --match   # both, in one process
```

Leave it running. It picks up those games by itself and plays each one with
your `create_agent()`. Nobody copies an id and nobody claims anything. If
several games are assigned at once, it plays all of them **at the same time**,
each in its own process with its own fresh agent instance. While it waits it
uses **no AI tokens**: it only asks the platform every ~10 seconds whether a
game is ready. Only playing a game with an LLM agent uses tokens.

**Werewolf:** `agent/agent.py` plays Werewolf with the strategy agent in
`agent/werewolf/` — see [`WEREWOLF.md`](WEREWOLF.md) for how it plays and
the offline simulator (`python -m sim.werewolf_sim`). For other games it
uses `examples.llm_agent` when `OPENAI_API_KEY` is set.

**The starter's original `agent/agent.py` was a placeholder.** It always plays the first
legal move. That finishes a Werewolf game, but it can't finish a Pokémon or
Red Alert match. Replace it with your own (see
[Writing your agent](#writing-your-agent)), or run the included LLM example,
which plays all three games (it needs `OPENAI_API_KEY` in `.env`):

```bash
python -m agent --check-tournament --agent examples.llm_agent
python -m agent --match --agent examples.llm_agent
```

Gameplay runs through the platform's generic MCP contract
(`get_game_state`/`wait_for_update`/`play_action`/...), so the runtime is
the same for Pokémon, Werewolf and Red Alert (see [`GAMES.md`](GAMES.md)):
only your `choose_action` needs to know how each game's moves look.

Step-by-step guide on the tournament site:
<https://platform.altruagent-game.com/tournament/agent-guide>

## Requirements

- Python 3.11+
- A UCLA tournament account with your event registration complete, and your
  agent set to **Self-hosted** in Agent Configuration. (Oracle-hosted play
  isn't available yet.)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # Windows (cmd): copy .env.example .env
```

## Quick start

1. **Get your key.** Sign in at
   <https://platform.altruagent-game.com/tournament/dashboard>, open
   **Agent Configuration**, and generate your Official Agent Key (`eak_live_...`). It's
   shown once, so put it in your `.env` right away:

   ```
   ALTRUAGENT_CONTROL_URL=https://api.altruagent-game.com
   ALTRUAGENT_OFFICIAL_AGENT_KEY=eak_live_...
   ```

   Keep it secret and never commit it. If it leaks, press **Rotate key** on
   the same page. The new key replaces the old one at once, so put it in
   `.env` and restart your agent right away. A process still running with
   the old key is turned away within about 10 seconds and stops; a game it
   was in the middle of can't continue from that process.

2. **Check your setup** (it plays nothing):

   ```bash
   python -m agent --check-tournament
   ```

   It checks that the control plane is reachable, that your key is accepted,
   that game assignments can be listed, and that your agent can be created.
   Every line should show `✓`; the last one is `✓ Ready to play Testing and
   tournament games`.

3. **Write your agent** in `agent/agent.py` (see
   [Writing your agent](#writing-your-agent)), or start from one of the
   `examples/`. The `agent/agent.py` you start with is a placeholder that
   always plays the first legal move: it finishes a Werewolf game, but it
   can't finish a Pokémon or Red Alert match. To use the LLM example instead,
   add `--agent examples.llm_agent` to the commands in steps 2 and 4.

4. **Run it and leave it running:**

   ```bash
   python -m agent --match                # test matches, while you try it out
   python -m agent --tournament           # your tournament games
   python -m agent --tournament --match   # both
   ```

## Configuration

| Variable | Required | Description |
|---|---|---|
| `ALTRUAGENT_CONTROL_URL` | yes | Base URL of the AltruAgent control plane. `.env.example` already sets the real deployed platform. |
| `ALTRUAGENT_OFFICIAL_AGENT_KEY` | yes | Your Official Agent Key (`eak_live_` + 64 hex characters), from the dashboard's Agent Configuration page. |
| `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `CLAUDE_EFFORT` | for Claude | Claude plays Werewolf (see `WEREWOLF.md`) and, through `examples/llm_agent.py`, Pokémon and Red Alert (default `claude-sonnet-5-5`, effort `low`). |
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | only for `examples/llm_agent.py` without Claude | See [Example LLM agent](#example-llm-agent). |

`.env` is loaded automatically and is already listed in `.gitignore` —
**never commit it**. Never print your key, or any token, in logs, error
messages, screenshots or commit messages; the runtime itself never does.

## Running your agent

Choose which games the process plays:

| Command | Plays |
|---|---|
| `python -m agent --tournament` | your **tournament games**: Swiss/bracket games, after you press *Register my agent* |
| `python -m agent --match` | your **test matches**: on the Testing page, matches you create with your seat set to *Mine (self-hosted)* or join from *Open matches* |
| `python -m agent --tournament --match` | both, in one process |

It prints `Connected with your Official Agent Key.`, which games it plays, and
then `Waiting for your next game...`. When a game is assigned to your agent it
prints what it picked up, plays it, and goes back to waiting:

```
Match assigned: werewolf (tournament)
  Tournament: Fall Cup, Swiss round 1 of 3
  Opponents: Alpha, Beta, Gamma, Delta, Epsilon, Zeta
  Connect by: 2026-10-16 15:04:00 UTC (3m 40s left)
Starting match...
```

The detail lines appear when the platform sends them: `(Testing)` or
`(tournament)`, the tournament and round, the other agents' names, and the
**connect deadline** with the time left. When the game ends, the game's own
line says how your agent did, for example
`finished: your agent (player 2) won (termination_reason=completed, score=1.0)`
(or `lost`, `drew`, `finished with no result` when the game couldn't be
finished), then the runtime prints `Match finished.` and goes back to waiting.

- **Test matches (`--match`).** On the dashboard's Testing page, create a test
  match and choose *Mine (self-hosted)* for the seats your agent should play,
  or join one from *Open matches*. A match with Open seats waits until other
  contestants fill them; once the last one is filled, your running `--match`
  process picks up each of your seats within about 10 seconds. If your agent
  plays several players in one match (self-play), each one is played in its
  own process. The runtime still prints one line for the whole match,
  `Match assigned: werewolf (Testing), self-play: your agent plays all 7 players`,
  and one `Match finished.` at the end; each player's own line says whether
  it won or lost.
- **Tournament games (`--tournament`).** Register your agent for a tournament
  on the dashboard. When a round starts, your games are assigned to your agent
  automatically. Keep the process running for the whole tournament.
- **A game of the other kind is left alone.** A `--match`-only process doesn't
  play tournament games. When one is waiting it warns you, once per game:
  `You have a tournament game waiting (Fall Cup, Swiss round 1 of 3): run with --tournament to play it — it counts as a loss if your agent doesn't connect within the window.`
  Start `python -m agent --tournament` before the deadline. A
  `--tournament`-only process notes a waiting test match with
  `Test match waiting: run with --match to play it`.
- **The connect deadline.** A game starts once every agent in it has
  connected. Your agent connects as soon as it picks the game up, so all you
  have to do is keep the process running. An agent that isn't connected by the
  deadline is a no-show and loses that game (see the tournament rules).
- **Several games at once.** Each game gets its own process and its own fresh
  `create_agent()` instance, so nothing leaks between games.
- **Choosing an agent.** `--agent MODULE[:FACTORY]` picks a factory other than
  `agent/agent.py`'s `create_agent()`, for example
  `python -m agent --tournament --agent examples.llm_agent` (a dotted module
  path importable from the repo root; `FACTORY` defaults to `create_agent`).
  Use the same `--agent` value for `--check-tournament`.
- **Reconnecting.** If the process stops mid-game, run the same command
  again. It signs in again, finds the game that is still assigned, and resumes
  it: at once if the old process stopped more than about 30 seconds ago,
  otherwise after up to about a minute. Until then it prints
  `Another runtime is playing this match with your Official Agent Key; checking again in 35s.`,
  because the stopped process's hold on the seat hasn't run out yet. Just
  leave it running. Nothing is saved locally.
- **One process with your key.** To play both kinds of game, run one process
  with `--tournament --match` rather than a `--match` process and a
  `--tournament` process side by side: the `--match` one would still warn
  about every tournament game, even one the other process is already playing.
  Don't run two copies with the same key either. Each of your agent's
  players in a game is played by whichever copy picks it up first; the other
  copy prints
  `Another runtime is playing this match with your Official Agent Key...` for
  it and leaves it alone. So nothing is played twice, and the game still
  finishes. But in a match where your agent plays several players
  (self-play), the two copies split the players: each terminal shows only
  part of the match. If the copies run different agents (say the placeholder
  and `--agent examples.llm_agent`), which agent plays which player is down
  to chance, and the match's result and history don't say.
- **Stopping.** Ctrl+C stops every game's process. It never resigns or
  otherwise touches a game.
- **If something goes wrong**, the message says what to do:
  - *The Official Agent Key was not accepted*: check that your agent is
    Self-hosted and that `.env` has the key exactly as it was shown. The
    dashboard can't show a key again: if you no longer have it, press
    **Rotate key** in Agent Configuration and put the new key in `.env`.
  - *Your Official Agent Key is no longer accepted, so no new game will
    start. Put your new key in .env ... and restart this process*: the key
    was rotated (or revoked) while the process was running. The platform
    turns the old key's sign-in away within about 10 seconds, so the process
    starts no new game, not even a test match, and stops. Put the new key in
    `.env` and restart it: the process reads `.env` only when it starts.
  - *Accept the updated Official Rules on your dashboard; I'll keep trying.*
    (or *Your event registration isn't complete. Finish it on your
    dashboard; I'll keep trying.*): do that on the dashboard. You don't need
    to restart: the process keeps running, checks again every 30 seconds and
    plays as soon as you have. Until then, games already running keep
    playing, but only for up to about an hour (until the process has to sign
    in again). New games may not start. So don't wait.
  - *Could not renew your agent session*: a temporary problem on the
    platform (it's busy, or briefly unreachable). Nothing to do: your running
    games keep playing and the process tries again by itself. It stops only
    for the first message above (the key itself was refused). (That's once
    it's running. If signing in fails when you start it, for any reason but
    the registration message above, it prints
    `Could not connect with your Official Agent Key: ...` and exits, so check
    that it printed `Connected with your Official Agent Key.` and run it
    again if not.)
  - *Connection problem (...); retrying for up to 90s.*: a dropped
    connection or a busy server during a game. Nothing to do: the game keeps
    going, the process tries again every few seconds, and your agent is asked
    again from the fresh state (a move is never sent twice blindly). It prints
    *Connection back; the game goes on.* when it's over.
  - *couldn't get into the game yet*: the same kind of problem before a game
    has started. The process keeps trying every few seconds until the connect
    deadline, so a short hiccup doesn't make you miss the game. After the
    deadline it prints *Couldn't reach that game (a temporary problem);
    retrying in 60s if it is still assigned.* and tries less often.
  - *Lost the connection to that game for a while*: the connection stayed
    down for 90 seconds during a game. The game is started again about 10
    seconds later, or after a longer pause if it keeps happening right after
    each restart. Nothing to do.
  - *The game server couldn't answer (...); retrying for up to 90s.*: the
    game server had a problem of its own while the process was reading the
    game. Nothing to do: it is retried like a connection problem and prints
    *The game server answers again; the game goes on.* when it's over.
  - *choose_action returned a move that can't be sent*: the move holds
    something that isn't plain JSON, such as a numpy number or an object of
    your own class. Use only `str`, `int`, `float`, `bool`, `None`, lists and
    dicts (for example `int(x)` for a numpy number). The game stops and is
    started again like after an agent error (see below).
  - *The game server couldn't handle that call (...); trying once more.*:
    the game server answered your move or message, but couldn't accept it
    (for example a move in a shape it doesn't take) or ran into a problem of
    its own. This isn't a connection problem, so your agent is asked once
    more. If it happens again, that game stops and is started again like
    after an agent error (next point).
  - *Your agent's process for this match stopped with an error (exit code 1);
    the match continues. Retrying in 60s if it is still assigned.*: your
    agent raised an error or made a move the game refused (the line just
    before it says what went wrong). The match itself goes on without your
    agent, and the game's own timers may play for it. The process is started
    again about a minute later if the match is still assigned. Your other
    games keep going. A game that keeps failing (for example one the game
    server lost after a restart, which the platform then closes with no
    result, or an agent that crashes on the same thing again) is retried less
    often each time: 1, 2, 4, 8, then every 10 minutes. After 10 minutes of
    play without trouble, the count starts over.
  - *Your red_alert match (Testing) has ended while your agent's process for
    it was stopped.*: the match ended before your agent was back in it, so it
    didn't play to the end.

**This is not a security sandbox.** Separate processes keep games apart from
*each other* (state, crashes). They don't isolate your agent code from your own
machine: it has whatever file and network access your user account has.

## Retired: the platform API key and Testing claim codes

AltruAgent now runs on the UCLA tournament site, and the Official Agent Key is
the only way an agent connects. Two older ways of connecting were turned off
on the platform:

| Retired | What happens now | Use instead |
|---|---|---|
| `python -m agent` with no mode, `ALTRUAGENT_API_KEY` (`sk_agent_...`) | prints a notice and exits | `python -m agent --tournament` with `ALTRUAGENT_OFFICIAL_AGENT_KEY` |
| `python -m agent --claim seatclaim_...`, `ALTRUAGENT_CLAIM_TOKEN` | prints `Testing claim codes were retired; run with --match and your Official Agent Key` and exits | `python -m agent --match` plays your test matches |
| `scripts/check_connection.py` | runs `python -m agent --check-tournament` | `python -m agent --check-tournament` |
| `scripts/check_sessions.py`, `scripts/check_tournaments.py` | print a notice | `--check-tournament` and the tournament dashboard |

The SDK classes behind them (`ApiKeyAuth`, `SeatGrantAuth`,
`client.sessions()`, `client.tournaments()`/`join_tournament()`,
`run_forever`, `run_forever_concurrent`) are still importable for reference,
but the platform answers them with HTTP 410; `ApiKeyAuth` and `SeatGrantAuth`
then raise an error with the same notice. The developer scripts
`scripts/smoke_game.py`, `scripts/acceptance_test.py` and
`scripts/check_game.py` used those APIs and no longer run against the
deployed platform.

## Writing your agent

See [`GAMES.md`](GAMES.md) for the supported games and their rules and
action formats.

This is the part you write, in `agent/agent.py`. The runtime looks for exactly
one name: `create_agent()` — a zero-argument factory, called once per game,
that returns your decision logic:

```python
def choose_action(state, context):
    return state.legal_actions[0]

def create_agent():
    return choose_action
```

That's the entire contract for a stateless agent — `create_agent()` just
hands back the plain function. **No base class, no decorator, no
registration.** `choose_action` is called only when it's actually that
game's turn (the runtime already checked) — pick one action from
`state.legal_actions` and return it. That is enough for Werewolf (where each
`action_id` is a seat number) and the Pokémon draft. It is **not** enough to
finish a Pokémon match or a Red Alert match: Pokémon's Team Preview and
doubles turns need a structured `dict`, and Red Alert has no
`legal_actions` (a move is a batch of orders). See [`GAMES.md`](GAMES.md).

`choose_action` may return any of:

- a `LegalAction` from `state.legal_actions` (the pattern above — for every
  game that lists its moves; not for Pokémon's Team Preview and doubles turns,
  or Red Alert)
- that `LegalAction`'s `action_id` (a `str`)
- a plain `int`, but **only** when it exactly matches one of the current
  legal actions' `action_id` as a string — this is what lets simple
  OpenSpiel-family agents just return `0`/`1`/etc.; it's rejected (never
  guessed) for a structured game whose `action_id`s aren't bare integers
- a structured `dict`, submitted as-is, for constructive actions that can't
  be enumerated as one of `state.legal_actions` (e.g. Pokémon's team
  submission, Red Alert's order batches) — this SDK performs no game-specific
  validation of it; the server is authoritative
- `altruagent.RESIGN`, to concede
- `altruagent.WAIT`, only in a real-time game (Red Alert): nothing to send
  right now; the runtime waits for the next view and asks again
- `altruagent.WithReasoning(<any move above>, "short public explanation")` —
  the same move, plus a `reasoning_summary` sent through `play_action` and
  shown to spectators next to the move (e.g. in GameHub). Keep it short and
  public; never put secrets in it.

Want per-game state? Return a fresh object instead of a bare function — the
runtime calling `create_agent()` again for the *next* game is what gives you a
new instance automatically:

```python
class MyAgent:
    def __init__(self):
        self.history = []
    def choose_action(self, state, context):
        self.history.append(state.move_count)
        ...

def create_agent():
    return MyAgent()
```

`create_agent()` may return a plain function or any object exposing a
callable `choose_action(self, state, context)`, optionally with the
`choose_message` and `on_action_result` methods described below. If the
object is itself callable (defines `__call__`), the runtime calls it directly
instead of its `choose_action`.

**Your `create_agent()` is called once per game, in that game's own process**
— never once for the whole run. Two games at once always get two separate
instances, so state kept on `self`, or even plain module-level variables,
never leaks between games. Deliberately sharing something *across* games (a
cache, a running total) needs your own external storage (a file, a database).

**Real-time games** (Red Alert; `state.raw["pacing"]["mode"] == "realtime"`)
use the same contract with three additions: `choose_action` may return
`WAIT`; a move the server refuses as a whole (`INVALID_ACTION`, usually
because units died between your read and your send) doesn't stop your agent —
the runtime re-reads the state and asks again; and `context.game_config`
holds the game's reference (rules, order formats, maps), fetched once per
game. An agent object may also define `on_action_result(self, result,
context)`: the runtime calls it after every move with the server's answer, or
with `{"error": code, "detail": message}` for a refusal it recovered from —
the only way to see a refused batch, since it never appears in a later state.
See [`GAMES.md`](GAMES.md#red-alert).

`context` (a `DecisionContext`) carries `session_id`, `game_type`,
`agent_id` (this seat's identity in the game), `seat_position` (your 0-based
seat) and `tournament_id` (set for a tournament game, `None` for a Testing
game) — enough to log or branch by game without parsing `state`. It
deliberately does **not** carry a client or session object — your decision
function can reason about the game, but can't accidentally act on another
one.

If your `choose_action` raises, returns something this SDK doesn't recognize,
or picks an action outside `state.legal_actions`, that one game's process
stops with a `DecisionError` and prints it, so a bug in your logic is visible
right away, and your other games keep going. The runtime starts that game
again about a minute later if it's still assigned, then less and less often
(see *If something goes wrong* above). Until your agent is back, the game's
own timers may play for you. A genuine server-side
race (a stale read producing `STALE_STATE`, or the game finishing between
your last read and your move) is handled automatically and never blamed on
your code.

**Ownership boundary:** the runtime owns signing in, finding your assigned
games, connecting to each one, running games concurrently, waiting
(long-polling) while it's not your turn, stopping cleanly if your agent is
eliminated mid-game (Werewolf), tracking `state_version`, and submitting your
move. Your code owns exactly two things: building your decision logic once per
game, and making the decision when asked.

### Messaging (Werewolf)

Some games have a messaging phase before or between moves — `state.phase ==
"messaging"` instead of the usual moving phase. You don't have to do
anything about this:
**if you don't define `choose_message`, your agent automatically votes to
end every messaging round it sees** and moves on — the same
`create_agent()`/`choose_action` contract above is already enough to
complete a messaging-enabled game.

If you want to actually talk, add an optional `choose_message` method next to
`choose_action` on the same object:

```python
from altruagent import SendMessage, TERMINATE_MESSAGING

class MyAgent:
    def choose_action(self, state, context):
        return state.legal_actions[0]

    def choose_message(self, state, context):
        for message in state.new_messages:   # everything said in this window so far (you get it again on every call)
            ...
        return SendMessage("let's cooperate")   # or: return TERMINATE_MESSAGING

def create_agent():
    return MyAgent()
```

- `SendMessage(content, recipients=None)` sends a chat message —
  `recipients=None`/`[]` broadcasts to everyone else; a single player index
  sends a private message (2+ recipients is rejected server-side today).
- `TERMINATE_MESSAGING` votes to end the round; once every active player has
  voted to end it, the phase flips back to moves.
- `state.new_messages` holds the whole current window (yours too) and is
  emptied when the phase changes, so it is empty when `choose_action` is
  asked for the day vote. Keep what you need from `choose_message` on `self`.
- `choose_message` is looked up the same way `choose_action` is (an
  attribute on whatever `create_agent()` returned) — **a plain function
  agent has no way to define one and just gets the default (auto-terminate)
  behavior.** Use a class-based agent (as above) if you want to talk.
- Word/message-count/length limits are enforced by the server, not this SDK;
  an invalid or over-quota `choose_message` result surfaces as a
  `DecisionError`, same as an invalid `choose_action` result. `state`
  doesn't expose your remaining quota — track your own usage if you need it
  (see `examples/messaging_agent.py`).
- Non-messaging games never touch any of this — `choose_message` is simply
  never called for them, whether or not you defined one.

See `examples/basic_agent.py` (moves only, relies on the default) and
`examples/messaging_agent.py` (a small stateful Werewolf talker) for two
complete, copy-pasteable starting points.

## Example agents

- `examples/basic_agent.py` — the plain contract: always the first legal
  action, like the placeholder in `agent/agent.py`. Finishes Werewolf only.
- `examples/messaging_agent.py` — a small stateful Werewolf agent that talks.
- `examples/smoke_agent.py` — valid, deterministic moves for Pokémon and
  Werewolf (no strategy; not Red Alert), handy for checking your setup end to
  end with a test match.
- `examples/llm_agent.py` — a general LLM agent that plays all three games
  (below).

Run any of them with `--agent`, for example
`python -m agent --match --agent examples.smoke_agent`.

### Example LLM agent

It's a reference, not a requirement — `agent/agent.py` can use any framework,
provider, or strategy you like.

`examples/llm_agent.py` is a general-purpose example agent: an OpenAI model
makes every decision, for any game, from what GameAPI supplies (the phase, your
seat's view of the state, recent messages, and the current legal options with
their instructions). It doesn't hard-code any game's rules. Set these in your
environment or in `.env`:

```
OPENAI_API_KEY=...          # required; never printed or logged
OPENAI_MODEL=gpt-4o-mini    # optional (default)
```

```bash
python -m agent --check-tournament --agent examples.llm_agent   # check it once
python -m agent --match --agent examples.llm_agent              # your test matches
python -m agent --tournament --agent examples.llm_agent         # your tournament games
```

- **Ordinary legal actions work for any game automatically.** When a game lists
  its moves, the model picks one exact `action_id` from the current legal
  actions. Werewolf (night actions and day votes) and Pokémon draft picks both
  work this way, and so will any future game that lists its moves.
- **Structured action templates need an adapter.** Some moves are a single
  template to fill in rather than a list; today that's Pokémon Team Preview and
  doubles turns. An adapter turns the template into bounded choices and checks
  the model's answer against the template's rules. The Pokémon adapter is in
  `examples/llm/pokemon.py`. A future structured game can add an adapter to
  `STRUCTURED_ADAPTERS` in `examples/llm_agent.py` without changing the rest of
  the agent. A template with no adapter stops the game with a clear error
  instead of guessing a payload, so not every future structured game works
  automatically.
- **Red Alert works too.** Red Alert is real time: no turns, and a move is a
  batch of orders sent whenever the agent is ready. The agent hands each Red
  Alert decision to its Red Alert player (`examples/llm/redalert.py`), a port of
  the platform's own Red Alert test agent: the model sees a compact view of the
  game (units, buildings, production, costs, visible enemies, the enemy's start
  cell and a ready-made attack order) and answers with a batch of orders in a
  strict format. Orders the server keeps refusing are fed back to the model,
  then dropped before sending while the reason still holds. When the model
  can't answer (an error, an unusable reply), nothing is sent for that moment:
  in real time a failing model simply acts less. Faster models act more often,
  so `OPENAI_MODEL` matters more here than in turn-based games.
- **Public reasoning:** each move carries the model's one-sentence public
  explanation (`WithReasoning`), sent as GameAPI's `reasoning_summary`.
- **In-game chat is separate from reasoning:** in a messaging phase (Werewolf
  discussion) the model may send a message or end the round, with at most 2
  model calls per discussion round.
- **Validation and fallback:** every answer is checked against the server's
  options. An invalid one is retried once with the reason, then replaced by a
  default legal action (logged as `FALLBACK`).
- **Pokémon's clocks:** battles run Showdown's VGC timer: 90 seconds at Team
  Preview, 55 seconds for each battle decision, and a 7-minute (420 s) total
  bank per player per battle. When a decision runs out, Showdown plays a
  default move for you and your bank shrinks. When your bank is empty, you
  forfeit the battle ("lost due to inactivity"). Each draft pick has 15
  seconds (see [`GAMES.md`](GAMES.md)). So for Pokémon the model gets at most
  40 seconds per battle decision or Team Preview and 10 seconds per draft
  pick, retry included (no single request over 25 seconds); then the agent
  plays its fallback. Werewolf and Red Alert have no such limit here.
- **No wasted calls:** the model is never called while you're waiting for
  another player or after the game ends.
- **Other providers:** the model provider is a small class (`examples/llm/providers.py`),
  so another provider can be added without touching the game logic.

## How the runtime works

You don't need this section to take part; it describes what
`python -m agent --tournament`/`--match` does under the hood.

1. **Sign in.** `OfficialAgentClient` (`altruagent/official.py`) exchanges
   your Official Agent Key for a short-lived agent session
   (`POST /tournament/agent/authenticate`). The key only ever goes to the
   control plane, never to GameAPI. When the session expires (about once an
   hour), the client signs in again and retries. If signing in again hits a
   temporary problem (too many attempts, a server error, a session the
   platform couldn't start), the running games keep playing and the
   supervisor tries again after a pause of 10 to 60 seconds (a full minute
   after "too many attempts"). If your registration isn't complete (for
   example the Official Rules were updated), it says what to do and tries
   again every 30 seconds. It stops only when the platform refuses the key
   itself. A rotated or revoked key also ends the sessions started with it:
   the next request (a poll or a lease renewal, within about 10 seconds) is
   refused, signing in again with the old key is refused too, and the
   process stops. If a game's worker is the first to find the key refused, it
   exits with its own code and the supervisor starts no new worker from then
   on, saying once what to do.
2. **Find games.** Every 10 seconds the supervisor
   (`altruagent/supervisor.py`, `run_tournament_forever`) lists your agent's
   active seats (`GET /tournament/agent/assignments`) and starts one worker
   process per seat of the kind it plays that doesn't have one. Each seat's
   `context` says its kind: `testing` (a test match, `--match`) or
   `tournament` (`--tournament`); a seat without one (an older backend) counts
   as a tournament game. A game of the other kind is left alone, with one
   warning or note per game. It logs each game it picks up
   (`describe_assignment`), once per match however many of its players it
   starts. A seat that drops off the list for two polls in a row has its
   worker stopped; the kind filter never stops a running worker. A game whose
   worker had stopped and that drops off the list is reported as ended. Each
   log line is written in one piece, so lines from several games never run
   together.
3. **Get a seat.** The worker (`altruagent/worker.py`) builds your agent, then
   asks for the seat's grant (`POST /tournament/agent/assignments/:seatId/grant`
   with this process's random `execution_id`), using the supervisor's agent
   session rather than signing in again. The grant holds a temporary
   GameAPI token for that one seat; it's kept only in memory and asked for
   again when it expires. A temporary failure is retried in the worker every
   few seconds for up to a minute; a worker that still can't get in is
   replaced at once while the game's connect deadline hasn't passed.
4. **Hold the seat.** While the game runs, the worker renews the seat's lease
   every 10 seconds (`.../lease/renew`). If another process takes the seat, the
   worker stops acting on it.
5. **Play.** The worker plays the game through `MCPGameSession`
   (`altruagent/mcp_game.py`) and `run_game` (`altruagent/runner.py`), calling
   your `choose_action`/`choose_message` when a decision is due, until the game
   ends. A call that fails for a temporary reason (a dropped connection, a
   gateway or server error, a busy game engine) is retried in place every few
   seconds; a move is never resent blindly, the state is read again instead.
   Only after 90 seconds without one successful call does the worker stop;
   the supervisor starts it again about 10 seconds later if the game had been
   playing, otherwise after the usual pause (1, 2, 4, 8, then every 10
   minutes). Each call gives up after 40 seconds without an answer, so a
   silently dropped connection costs seconds, not minutes. A move or message
   the game server answers with an MCP tool error (it couldn't accept the
   arguments, or the tool failed) is not a connection problem: it is tried
   once more, then the worker stops. A read answered that way is the
   server's own trouble and is retried like a connection problem. A move or
   message that can't be sent as JSON stops the worker at once, as an agent
   error.

### Playing one game by hand

`MCPGameSession` is a handle to one game, played through the platform's
generic MCP gameplay contract. You won't normally call it yourself — the
runtime already does, including tracking `state_version` — but it's the piece
to look at if you want to experiment:

```python
state = game.get_state()                      # is it my turn? what phase? + legal_actions if so
result = game.play_action(action_id=state.legal_actions[0].action_id, state_version=state.state_version)
state = game.wait_for_update(since_version=state.state_version)  # returns as soon as anything changes
result = game.resign()
```

`state.legal_actions` is a list of `LegalAction`s (`action_id`, `label`,
`input`, `raw`) — the server includes them only when you can act
(`state.is_current_actor`), and it's empty otherwise. Always re-check it on
the latest state rather than assuming; the server is the authority and will
reject a stale or invalid action.

## Project layout

```
altruagent/     # SDK — hides HTTP/auth plumbing. You shouldn't need to edit this.
agent/          # Your agent code goes here. __main__.py is `python -m agent`'s entry point.
examples/       # Copy-pasteable starting points for agent/agent.py.
scripts/        # check_connection.py (= --check-tournament) and retired developer scripts.
tests/          # Unit tests for the SDK, run against mocked HTTP responses.
```

## Running tests

```bash
pytest
```

Tests use mocked HTTP responses and do not require network access or a real
platform account.
