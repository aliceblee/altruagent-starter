# Red Alert: team playbook (edit me)

Sent with every Red Alert decision. Plain English; change the strategy here,
no code needed. Keep it short.

## How you win

- Destroying everything wins outright, but most matches hit the time limit.
  Then the tiebreak is **kills_cost - deaths_cost**, then **assets_value**: value
  you destroyed minus value you lost, then what you still own. So **don't throw
  units away** in hopeless attacks, but **never stop building an army**:
  losing your base loses the game outright. In testing we "held a few riflemen
  to protect the tiebreak" while the opponent massed tanks, and lost our base
  twice in a row.

## Economy first (test games: we rushed infantry, the refinery came last, we ran out of money)

1. Deploy the MCV.
2. Power plant (powr), then **refinery (proc) right away** (it brings a
   harvester), then barracks (tent/barr), then a **second refinery**.
3. War factory (weap), placed by about 100 s. Until it is placed, train at
   most 4-5 riflemen: the money goes to the war factory and refineries first.
   (In testing we trained infantry nonstop from the barracks and the war
   factory wasn't placed until 156-240 s; the first tank came at 2-6 minutes.)
4. **The war factory's first unit is a harvester (harv)**, then tanks. Keep
   about two harvesters per refinery all game (in 9 of 13 test games we never
   trained a single extra harvester).
5. Keep power ahead of demand: add a power plant before the war factory and
   whenever power_drained gets close to power_provided. Low power slows every
   queue.
- Never queue a building you already have in the queue or ready to place, and
  **never cancel a building that's in progress**: cancelling throws its
  progress away (the code blocks cancelling the only copy). Place finished
  buildings at once (place without `at`).
- Keep harvesters harvesting; if one is idle or under attack, send it back to
  the ore and defend it.
- **Your money is cash plus stored ore.** In OpenRA, ore sitting in refineries
  and silos is spent automatically when you build, so "cash 0" with ore
  stored is not broke: keep queuing. If ore storage is full, harvesters can't
  unload and income stops: build a silo (or spend) at once. In testing we
  sat at "cash is zero, waiting for income" for 7 minutes with ore stored.
- Rebuild a lost harvester before anything else. One harvester can't feed a
  war factory (in testing we had "the lone harvester" for most of a match and
  trained 2 tanks in 10 minutes).

## Army

- **Production never stops.** From the moment the barracks and war factory
  exist, keep both queues busy for the whole game (queue the next unit before
  the queue runs empty). Spend most money on tanks from the war factory.
- Aim for 8+ combat units by 3 minutes and keep growing. "Hold and defend"
  means keep producing at home, never idle with a handful of units.
- Build 1-2 defenses (pillbox / turret / tesla coil / flame tower, whatever
  available_production offers) once the war factory is up, between your base
  and the enemy, and more when under_attack events show a bigger enemy army.

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
