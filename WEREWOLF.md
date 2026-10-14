# Werewolf agent

`python -m agent --tournament --match` now plays Werewolf with
`agent/werewolf/` (other games fall back to `examples.llm_agent` when
`OPENAI_API_KEY` is set, else the first legal move).

## Claude as the brain (default when `ANTHROPIC_API_KEY` is set)

With `ANTHROPIC_API_KEY` in `.env`, Claude (`claude_brain.py`, default model
`claude-opus-5-5`) makes every Werewolf decision: night kill, seer check,
day vote, and every chat, including private wolf-to-wolf messages. For each
decision it gets:

- what the server shows our seat: role, partner, seer results, the living and
  the dead with their revealed roles, the vote history, the whole chat;
- the **team playbook**, `agent/werewolf/playbook.md`. **Change strategy
  here**: it's plain English, no code;
- its own private notes from earlier decisions (plans, reads, a fake-claim
  story), so it stays consistent through the game;
- the rule engine's analysis below: exact P(wolf) / P(seer) per seat, seer
  claims, unison pushes, and the move the rules would make, as a hint.

The rule engine is also the safety net. If Claude errors, times out (40 s for
moves; chat stays inside the 2-minute window), refuses, or picks something
illegal, the rules' move is played. One team rule is enforced in code as
well: the seer never claims on day 1. Every Claude call is logged (latency,
tokens, fallbacks) as `note` lines in `logs/decisions/<session>/seat<N>.jsonl`.

Knobs (`.env`): `WEREWOLF_CLAUDE_MODEL` (`claude-sonnet-5-5` or
`claude-haiku-5-5` are faster and cheaper), `WEREWOLF_CLAUDE_EFFORT` (votes
and nights, default `medium`), `WEREWOLF_CLAUDE_CHAT_EFFORT` (default
`low`), `WEREWOLF_BRAIN=rules` to switch Claude off.

Compare the two brains at one test table with `examples/mixed_table.py`
(seat kinds `ours_claude` and `ours_rules`).

## How the rule engine plays

**Beliefs (`beliefs.py`).** With 7 seats, 2 wolves and 1 seer there are only
105 possible role assignments, so the agent enumerates all of them, drops the
ones that contradict hard facts (roles revealed at death, its own role, wolf
partner, seer results) and weights the rest by soft evidence: seer claims and
their announced results, votes, accusations and defenses in chat, and who the
wolves chose to kill. The result is an exact P(wolf) and P(seer) per seat.
The weights in `Weights` are the tuning knobs.

**Strategy (`agent.py`).**

| Role | Night | Day vote | Chat |
|---|---|---|---|
| Villager | – | Highest P(wolf); never abstains; breaks near-ties toward the table's announced target | Announces its vote early and gives a reason; asks the seer to claim on day 1; backs a credible lone seer claim; defends itself when accused |
| Seer | Checks rival seer claimants first, then the likeliest wolf | A found wolf, else like a villager | Claims as soon as it finds a wolf, on day 2 at the latest, at once if someone else claims seer or if it comes under pressure |
| Wolf | Kills seer claimants first, then likely seers and trusted villagers, sparing players who accused it (their death would point at it); follows its partner's private suggestion | The villager the *public* evidence points at, joining any momentum; spreads its vote when the lynch is already decided; with 5 or fewer alive, steers the announced tally to a villager or a tie, never a wolf; votes out its partner only when the partner is doomed anyway | Plays villager; on day 1 voices mild doubt about its partner; whispers the plan to its partner (varied wording, stops if the partner never answers); backs a partner's fake claim; counter-claims seer when a claimant outs it |

Why never abstain: with 7 players a wrong lynch on both day 1 and day 2
hands the wolves parity, and 3 days in a row with no elimination is also a
wolf win. A split or abstaining village is the wolves' best friend.

**Chat (`chat_llm.py`).** The rules decide *what* to say. If
`OPENAI_API_KEY` is set, an LLM rewrites the message to sound more natural.
It is never told the agent's real role, and a rewrite that drops the vote
target, names a new player, or adds or drops a seer claim is thrown away.
Turn it off with `WEREWOLF_LLM_CHAT=0`. Messages are capped at 45 words and 4
per day (the server allows 50 words and 5).

## Offline testing

```bash
python -m sim.werewolf_sim --games 2000             # benchmark (~70 s)
python -m sim.werewolf_sim --games 50 --verbose 2   # read two full game transcripts
```

`sim/werewolf_sim.py` re-implements the platform rules from GAMES.md and
feeds agents GameAPI-shaped states. Games are paired: game *g* deals the
same roles and seats whichever agent is being tested, so the two rows in each
block are directly comparable.

Current results (2000 paired games each, the seat our agent holds):

| Opponents (6 of them) | Their seat | Our seat |
|---|---|---|
| heuristic (seer claims, villagers follow it) | 48.5% | **58.5%** (villager 55.2, seer 51.6, wolf 67.6) |
| random | 34.6% | **36.8%** |
| first_legal (starter placeholder) | 47.9% | 45.1% |

The `first_legal` table is degenerate: its wolves always kill the
lowest-numbered villager, which leaks who the wolves are to its own
"lowest seat" voting. It doesn't reflect real opponents.

**When accused.** The agent answers each new accuser once ("I'm a villager.
PlayerX pushing me without evidence is exactly what a wolf would do."). When
announced votes against it reach at least 2 and come within one of half the
other voters, it sends one *rescue* message a day rallying everyone onto its
top suspect (`WerewolfAgent.rescue`). A seer accused by 2+ players claims. An
optional belief weight, `Weights.accused_me_falsely`, makes a known-innocent
agent suspect its accusers; it is off (1.0) because it cost the village 1-6
points in the simulator.

Findings from ablations (in the commit history): a seer claiming on day 1
*without* a wolf found loses about 10 points; a wolf counter-claiming seer
only when *it* is outed beats always and never.

## Before the tournament: verify the observation format

The exact wording of the private observation (role, wolf partner, seer
results) isn't documented in this starter, so `parsing.py` reads it
tolerantly. The agent also infers its role and partner from the night
actions it's offered, and prints a `WARNING` if it can't read its role.
Run one test match with debug logging and check the first lines:

```bash
WEREWOLF_DEBUG=1 python -m agent --match
# [werewolf] first observation: '...' role=seer partners=set()
```

If role or partners or seer results come out wrong, paste that observation
line into `tests/test_werewolf_agent.py` as a new case and adjust the
patterns in `parsing.py`.
