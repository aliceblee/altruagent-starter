# Red Alert: team playbook (edit me)

Sent with every Red Alert decision. Plain English; change the strategy here,
no code needed. Keep it short.

## How you win

- Destroying everything wins outright, but most matches hit the time limit.
  Then the tiebreak is **kills_cost - deaths_cost**, then **assets_value**: value
  you destroyed minus value you lost, then what you still own. So **don't throw
  units away**: a lost attack is points for the opponent, and a big, intact
  army and base count at the end.

## Economy first (test games: we rushed infantry, the refinery came last, we ran out of money)

1. Deploy the MCV.
2. Power plant (powr), then **refinery (proc) right away** (it brings a
   harvester), then barracks (tent/barr), then a **second refinery**.
3. War factory (weap), then more harvesters (harv) until each refinery has
   about two. Add power whenever power_drained is close to power_provided.
- Never queue a building you already have in the queue or ready to place;
  place finished buildings at once (place without `at`).
- Keep harvesters harvesting; if one is idle or under attack, send it back to
  the ore and defend it.
- Money at zero means too few harvesters: build a harvester (or a refinery)
  before more units. (Test games won, but stalled at zero cash mid-game.)

## Army

- After the economy is running, build steadily from barracks and war factory:
  tanks (1tnk/2tnk/3tnk, from available_production) are worth far more than
  rifle infantry (e1); mix in rocket infantry (e3) against vehicles.
- Rally new units to a spot **in front of your base**, not into the enemy.
- Don't trickle units into the enemy one by one. Attack with one large group
  (roughly 10+ units including tanks), attack-moving together, and target
  harvesters, refineries and power plants first.
- If an attack is losing (units dying, enemy defenses), pull the survivors
  back to base: saving them is worth points.
- When under_attack events arrive, send everything nearby to defend; repair
  damaged buildings when you have money.
- With a few minutes left and ahead on the tiebreak: defend and keep what you
  have. Behind: attack weak, valuable targets (harvesters) with your full army.
