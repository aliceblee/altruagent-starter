"""Exact Bayesian reasoning over every possible role assignment.

Seven seats, two wolves and one seer give only 21 x 5 = 105 possible worlds,
so instead of hand-tuned suspicion points we enumerate them all, drop the
ones that contradict hard facts (revealed roles, my own role and private
info), weight the rest by how well they explain the soft evidence (seer
claims, votes, accusations), and read off each seat's probability of being a
wolf or the seer.

The soft-evidence factors are the knobs to tune with the simulator
(``sim/werewolf_sim.py``): each one says how much more (or less) likely an
observation is in a world where the given players are wolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .parsing import SEER, VILLAGER, WOLF, ChatFacts, Snapshot

N_WOLVES = 2


@dataclass(frozen=True)
class Weights:
    # A seer claim from a wolf (fake claim) vs. from the real seer vs. from a plain villager.
    claim_by_wolf: float = 0.15
    claim_by_villager: float = 0.04
    # A real seer's announced result that contradicts the world.
    false_result_by_seer: float = 0.01
    # A "seer" who only claimed after another claimant named them a wolf, and whose
    # result names that accuser a wolf. A real seer checks one of six players a
    # night, so that coincidence is rare; a cornered wolf does it every time.
    reactive_counterclaim_as_seer: float = 0.25
    # A villager fake-claiming seer and "finding" a non-wolf to be a wolf.
    false_wolf_by_villager_claimant: float = 0.3
    # A fake seer "finding" their own partner to be a wolf.
    wolf_claims_partner_wolf: float = 0.15
    # A wolf voting for / accusing its own partner.
    wolf_votes_partner: float = 0.35
    wolf_accuses_partner: float = 0.55
    # A wolf clearing / defending its partner.
    wolf_clears_partner: float = 1.6
    # Someone the night's victim had accused or voted for (wolves silence their accusers).
    accused_by_victim: float = 1.5
    # Voting for a player later revealed as a villager / wolf.
    wolf_voted_villager_lynch: float = 1.25
    wolf_voted_wolf_lynch: float = 0.6
    # Someone accusing or voting for me when I know I'm not a wolf: they're either
    # mistaken or a wolf. Counted once per accuser per day (chat or vote).
    # Off (1.0) by default: in the simulator any value above 1 cost the village
    # 1-6 points, because innocent villagers who suspect each other then split
    # their votes instead of converging. Try 1.2-1.4 against real opponents.
    accused_me_falsely: float = 1.0
    # Two players pushing the same target in unison, unprompted (see parsing.ChatFacts.unison).
    unison_push: float = 1.5  # simulator: call-out alone +6 pts village wins, with 1.5 +11 (2.0: +15, likely flattered)


DEFAULT_WEIGHTS = Weights()


@dataclass
class Beliefs:
    p_wolf: dict[int, float]
    p_seer: dict[int, float]
    worlds: int

    def top_wolf(self, candidates: list[int]) -> list[int]:
        return sorted(candidates, key=lambda s: (-self.p_wolf.get(s, 0.0), s))


def infer(
    snap: Snapshot,
    chat: ChatFacts,
    *,
    private: bool = True,
    weights: Weights = DEFAULT_WEIGHTS,
) -> Beliefs:
    """Probabilities from this seat's point of view (``private=True``) or from
    what any outside observer knows (``private=False``; used by wolves to
    predict what the villagers will believe)."""
    w = weights
    seats = list(range(snap.n_players))
    known: dict[int, str] = {s: d.role for s, d in snap.dead.items() if d.role}
    if private and snap.role:
        known[snap.me] = snap.role
        if snap.role == WOLF:
            for p in snap.partners:
                known[p] = WOLF
        if snap.role == SEER:
            for target, verdict in snap.seer_results.items():
                if verdict == WOLF:
                    known[target] = WOLF
                elif target not in known:
                    known[target] = VILLAGER  # "not a wolf": a plain villager, since I'm the seer
        if snap.role == VILLAGER:
            pass

    night_victims = [d for d in snap.dead.values() if d.cause and "night" in d.cause.lower()]
    lynched = {d.seat: d for d in snap.dead.values() if d.cause and "vote" in d.cause.lower()}

    victim_accused: dict[int, set[int]] = {}
    for victim in night_victims:
        # Only accusations the victim started: hitting back at someone who had already
        # accused them is retaliation, not a read (in testing a seer killed after answering
        # a random day-1 pick framed that innocent picker).
        targets: set[int] = set()
        for i, (s, t, day) in enumerate(chat.accusations):
            if s == victim.seat and not any(a == t and b == victim.seat and d == day for a, b, d in chat.accusations[:i]):
                targets.add(t)
        targets |= {v.target for v in snap.votes if v.voter == victim.seat and v.target is not None
                    and not any(a == v.target and b == victim.seat and d == v.day for a, b, d in chat.accusations)}
        victim_accused[victim.seat] = targets

    # Who has accused or voted for me, when I know I'm innocent: (accuser, day) pairs.
    false_accusers: list[int] = []
    if private and snap.role in (VILLAGER, SEER):
        pairs = {(s, d) for s, t, d in chat.accusations if t == snap.me}
        pairs |= {(v.voter, v.day) for v in snap.votes if v.target == snap.me}
        false_accusers = [s for s, _ in pairs if s != snap.me]

    totals_w = {s: 0.0 for s in seats}
    totals_s = {s: 0.0 for s in seats}
    z = 0.0
    count = 0
    for wolves in combinations(seats, N_WOLVES):
        wset = set(wolves)
        for seer in seats:
            if seer in wset:
                continue
            if not _consistent(known, wset, seer):
                continue
            weight = 1.0
            for claimant in chat.seer_claims:
                if claimant in wset:
                    weight *= w.claim_by_wolf
                    for target, verdict in chat.claimed_results.get(claimant, {}).items():
                        if verdict == WOLF and target in wset:
                            weight *= w.wolf_claims_partner_wolf
                elif claimant != seer:
                    weight *= w.claim_by_villager
                    for target, verdict in chat.claimed_results.get(claimant, {}).items():
                        if verdict == WOLF and target not in wset:
                            weight *= w.false_wolf_by_villager_claimant
                else:
                    for target, verdict in chat.claimed_results.get(claimant, {}).items():
                        if (verdict == WOLF) != (target in wset):
                            weight *= w.false_result_by_seer
                    results = chat.claimed_results.get(claimant, {})
                    if any(results.get(a) == WOLF for a in chat.named_before_claim.get(claimant, ())):
                        weight *= w.reactive_counterclaim_as_seer
            for vote in snap.votes:
                if vote.target is None:
                    continue
                if vote.voter in wset and vote.target in wset:
                    weight *= w.wolf_votes_partner
                if vote.voter in wset and vote.target in lynched and vote.day == lynched[vote.target].day:
                    role = lynched[vote.target].role
                    weight *= w.wolf_voted_wolf_lynch if role == WOLF else w.wolf_voted_villager_lynch
            for speaker, target, _ in chat.accusations:
                if speaker in wset and target in wset:
                    weight *= w.wolf_accuses_partner
            for speaker, target, _ in chat.clears:
                if speaker in wset and target in wset:
                    weight *= w.wolf_clears_partner
            for first, second, _, _ in chat.unison:
                if first in wset and second in wset:
                    weight *= w.unison_push
            for accuser in false_accusers:
                if accuser in wset:
                    weight *= w.accused_me_falsely
            for targets in victim_accused.values():
                for t in targets:
                    if t in wset:
                        weight *= w.accused_by_victim
            z += weight
            count += 1
            for s in wset:
                totals_w[s] += weight
            totals_s[seer] += weight

    if z <= 0:
        # Contradictory evidence (e.g. a misread observation): fall back to uniform over the unknown.
        unknown = [s for s in seats if s not in known]
        share = N_WOLVES / max(1, len(unknown))
        return Beliefs({s: (1.0 if known.get(s) == WOLF else share if s in unknown else 0.0) for s in seats},
                       {s: 0.0 for s in seats}, 0)
    return Beliefs({s: totals_w[s] / z for s in seats}, {s: totals_s[s] / z for s in seats}, count)


def _consistent(known: dict[int, str], wolves: set[int], seer: int) -> bool:
    for seat, role in known.items():
        if role == WOLF and seat not in wolves:
            return False
        if role == SEER and seat != seer:
            return False
        if role == VILLAGER and (seat in wolves or seat == seer):
            return False
    return True
