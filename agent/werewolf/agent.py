"""A Werewolf agent: Bayesian beliefs, role-specific play, and table talk.

Decisions (night kills, investigations, votes) are always made by the rules
below, so the agent is fast, deterministic enough to test, and never stalls
on a model call. Chat content is planned by the same rules (what to claim,
who to accuse, who to vote for); an optional LLM (see ``chat_llm.py``) only
rewrites the planned message to sound more natural and persuasive.

Key points of the strategy, for 7 players / 2 wolves / 1 seer:

- The village must hit a wolf with one of its first two lynches (a wrong
  lynch on day 1 and day 2 hands the wolves parity), and a day with no
  lynch helps the wolves (three in a row is a wolf win). So non-wolves
  never abstain, announce their vote early to concentrate the plurality,
  and break near-ties toward the target the table is already converging on.
- The seer claims as soon as it finds a wolf, on day 2 at the latest, and
  immediately if someone else claims seer.
- Wolves vote with the village's plausible choice (computed from public
  information only), kill seer claimants first, counter-claim seer when a
  claimant accuses one of them, and coordinate privately with the partner.
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState, SendMessage

from .beliefs import Beliefs, infer
from .parsing import ABSTAIN, SEER, VILLAGER, WOLF, ChatFacts, Resolver, Snapshot, read_message, snapshot

MAX_CHATS_PER_DAY = 4  # the server allows 5; keep one in reserve
MAX_WORDS = 45  # the server's cap is 50
MIN_SECONDS_TO_SPEAK = 6.0
BANDWAGON_WEIGHT = 0.06  # per announced vote, when picking among near-equal suspects
# Our first message of day 1, sent word for word whatever our role (never LLM-polished).
DAY1_OPENER = "I'm a vegetarian vampire villager Eddie Cullen. I hate werewolves. Raise your hand if you're werewolf?"


def _p(seat: int) -> str:
    return f"Player{seat}"


class WerewolfAgent:
    # Strategy switches (also used by the simulator for ablations).
    counter_claim = "self"  # True: when either wolf is outed; "self": only when I am; False: never
    seer_claim_day = 2  # claim on this day at the latest, even without a wolf found

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        polish: Callable[[str, dict], str] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.polish = polish
        self.log = log or (print if os.environ.get("WEREWOLF_DEBUG") else (lambda _msg: None))
        self.chat = ChatFacts()
        self._seen: set[int] = set()
        self._logged_observation = False
        # Per-day chat bookkeeping.
        self._chat_day: int | None = None
        self._sent_today = 0
        self._done_today = False
        self._said: set[str] = set()  # topics already covered today
        self._sent_opener = False
        self._reacted_claims: set[int] = set()
        self._reacted_accusers: set[int] = set()
        self._announced_target: int | None = None
        # Seer state.
        self._claimed_seer = False
        # Wolf state.
        self._fake_seer = False
        self._fake_results: dict[int, str] = {}
        self._told_partner_day: int | None = None
        # Facts learned from the legal actions offered (robust to observation wording).
        self._known_role: str | None = None
        self._known_partners: set[int] = set()

    # -- runtime hooks ----------------------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        snap, resolver = self._update(state, context)
        legal = {a.action_id: a for a in state.legal_actions}
        seats = sorted(int(a) for a in legal if a.isdigit() and int(a) != ABSTAIN)
        if not seats:
            return state.legal_actions[0]
        phase = snap.sub_phase or _infer_sub_phase(state)
        if phase == "night_wolf":
            choice = self._wolf_kill(snap, seats)
        elif phase == "night_seer":
            choice = self._seer_check(snap, seats)
        else:
            choice = self._day_vote(snap, seats)
        self.log(f"[werewolf] P{snap.me} ({snap.role}) {phase}: -> {choice}")
        return legal.get(str(choice), state.legal_actions[0])

    def choose_message(self, state: GameState, context: DecisionContext):
        snap, resolver = self._update(state, context)
        if self._chat_day != snap.day:
            self._chat_day, self._sent_today, self._done_today = snap.day, 0, False
            self._said = set()
            self._announced_target = None
        if self._done_today or self._sent_today >= MAX_CHATS_PER_DAY:
            self._done_today = True
            return TERMINATE_MESSAGING
        if snap.seconds_left is not None and snap.seconds_left < MIN_SECONDS_TO_SPEAK:
            self._done_today = True
            return TERMINATE_MESSAGING

        if snap.day <= 1 and not self._sent_opener:
            self._sent_opener = True
            self._sent_today += 1
            self.log(f"[werewolf] P{snap.me} says: {DAY1_OPENER}")
            return SendMessage(DAY1_OPENER)

        private = self._private_message(snap)
        if private is not None:
            self._sent_today += 1
            return private

        plan = self._plan_message(snap)
        if plan is None:
            self._done_today = True
            return TERMINATE_MESSAGING
        text, facts = plan
        if self.polish is not None:
            try:
                text = self.polish(text, facts) or text
            except Exception as exc:  # noqa: BLE001 - never let chat polish break the game
                self.log(f"[werewolf] chat polish failed: {exc!r}")
        text = " ".join(text.split()[:MAX_WORDS])
        self._sent_today += 1
        self.log(f"[werewolf] P{snap.me} says: {text}")
        return SendMessage(text)

    # -- state ------------------------------------------------------------------------------

    def _update(self, state: GameState, context: DecisionContext) -> tuple[Snapshot, Resolver]:
        snap, resolver = snapshot(state, context.seat_position)
        read_role = snap.role
        self._fill_from_legal_actions(snap, state)
        if not self._logged_observation:
            self._logged_observation = True
            self.log(f"[werewolf] first observation: {state.observation!r} role={snap.role} partners={snap.partners}")
            if read_role is None:
                print(
                    "[werewolf] WARNING: couldn't read this seat's role from the observation "
                    f"(using {snap.role or 'villager'} for now; night actions correct it). "
                    f"Observation was: {state.observation!r}",
                    flush=True,
                )
        for message in state.new_messages:
            if message.index in self._seen or message.type != "chat":
                continue
            self._seen.add(message.index)
            read_message(self.chat, message.sender, message.content, list(message.recipients), snap.day, snap.me, resolver)
        return snap, resolver

    def _fill_from_legal_actions(self, snap: Snapshot, state: GameState) -> None:
        """The night actions themselves reveal role and partners, whatever the
        observation's wording: only wolves get ``night_wolf`` turns, and their
        targets are exactly the living non-wolves."""
        if state.legal_actions and state.phase != "messaging":
            phase = snap.sub_phase or _infer_sub_phase(state)
            if phase == "night_wolf":
                self._known_role = WOLF
                targets = {int(a.action_id) for a in state.legal_actions if a.action_id.isdigit()}
                self._known_partners |= {s for s in snap.alive if s != snap.me and s not in targets}
            elif phase == "night_seer":
                self._known_role = SEER
        if snap.role is None and self._known_role is not None:
            snap.role = self._known_role
        if snap.role == WOLF:
            snap.partners |= self._known_partners
            self._known_partners |= snap.partners

    def _beliefs(self, snap: Snapshot, *, private: bool = True) -> Beliefs:
        return infer(snap, self.chat, private=private)

    # -- night ------------------------------------------------------------------------------

    def _seer_check(self, snap: Snapshot, seats: list[int]) -> int:
        unchecked = [s for s in seats if s not in snap.seer_results] or seats
        beliefs = self._beliefs(snap)
        # Prefer seer claimants (resolve the claim war), then the likeliest wolf;
        # random among equals so wolves can't predict us.
        claimants = [s for s in unchecked if s in self.chat.seer_claims]
        if claimants:
            return max(claimants, key=lambda s: beliefs.p_wolf.get(s, 0))
        best = max(beliefs.p_wolf.get(s, 0) for s in unchecked)
        top = [s for s in unchecked if beliefs.p_wolf.get(s, 0) >= best - 1e-9]
        return self.rng.choice(top)

    def _wolf_kill(self, snap: Snapshot, seats: list[int]) -> int:
        targets = [s for s in seats if s not in snap.partners]
        if not targets:
            return seats[0]
        # A partner's private suggestion wins (ties go to the lower seat anyway).
        suggestion = self._partner_suggestion(snap, targets)
        if suggestion is not None:
            return suggestion
        return self._kill_ranking(snap, targets)[0]

    def _kill_ranking(self, snap: Snapshot, targets: list[int]) -> list[int]:
        public = self._beliefs(snap, private=False)
        wolves = snap.partners | {snap.me}

        def score(t: int) -> float:
            s = 0.0
            if t in self.chat.seer_claims:
                s += 5.0  # a live seer claimant is the biggest threat
            s += 3.0 * public.p_seer.get(t, 0.0)
            s += 0.8 * sum(1 for spk, tgt, _ in self.chat.accusations if spk == t and tgt in wolves)
            s += 0.6 * sum(1 for v in snap.votes if v.voter == t and v.target in wolves)
            # Players the village trusts carry weight in discussion; a cleared player
            # is also a confirmed villager, which makes them a strong voice.
            s += 1.0 * (1.0 - public.p_wolf.get(t, 0.0))
            s += 0.5 * sum(1 for _, tgt, _ in self.chat.clears if tgt == t)
            return s + self.rng.random() * 0.05

        return sorted(targets, key=score, reverse=True)

    def _partner_suggestion(self, snap: Snapshot, targets: list[int]) -> int | None:
        for sender, text, day in reversed(self.chat.private_from):
            if sender in snap.partners and day == snap.day:
                lowered = text.lower()
                if "kill" not in lowered and "tonight" not in lowered and "target" not in lowered:
                    continue
                for seat in targets:
                    if _p(seat).lower() in lowered.replace(" ", ""):
                        return seat
        return None

    # -- day vote ---------------------------------------------------------------------------

    def _day_vote(self, snap: Snapshot, seats: list[int]) -> int:
        if snap.role == WOLF:
            return self._wolf_vote(snap, seats)
        return self._village_vote(snap, seats)

    def _village_vote(self, snap: Snapshot, seats: list[int]) -> int:
        beliefs = self._beliefs(snap)
        if snap.role == SEER:
            known_wolves = [s for s in seats if snap.seer_results.get(s) == WOLF]
            if known_wolves:
                return known_wolves[0]
        intents = self.chat.intents_on(snap.day)

        def score(t: int) -> float:
            backers = sum(1 for spk, tgt in intents.items() if tgt == t and beliefs.p_wolf.get(spk, 0) < 0.5)
            return beliefs.p_wolf.get(t, 0.0) + BANDWAGON_WEIGHT * backers

        best = max(score(s) for s in seats)
        top = [s for s in seats if score(s) >= best - 1e-9]
        if self._announced_target in top:
            return self._announced_target
        return min(top)  # deterministic, so like-minded villagers converge

    def _wolf_vote(self, snap: Snapshot, seats: list[int]) -> int:
        partners = snap.partners & snap.alive
        intents = self.chat.intents_on(snap.day)
        alive = len(snap.alive)
        for p in partners:
            against = sum(1 for spk, tgt in intents.items() if tgt == p and spk != snap.me)
            if against * 2 > alive - 1 and p in seats:
                return p  # the partner is going anyway: join in and look like a villager
        villagers = [s for s in seats if s not in partners]
        if not villagers:
            return seats[0]
        return self._wolf_preferred_lynch(snap, villagers)

    def _wolf_preferred_lynch(self, snap: Snapshot, villagers: list[int]) -> int:
        public = self._beliefs(snap, private=False)
        intents = self.chat.intents_on(snap.day)

        def score(t: int) -> float:
            backers = sum(1 for spk, tgt in intents.items() if tgt == t and spk != snap.me)
            s = public.p_wolf.get(t, 0.0) + 0.15 * backers
            if t in self.chat.seer_claims:
                s += 0.6  # a real seer is the best possible lynch
            if self._fake_seer and self._fake_results.get(t) == WOLF:
                s += 1.0  # stay consistent with our fake claim
            return s

        if self._announced_target in villagers:
            return self._announced_target
        return max(villagers, key=lambda t: (score(t), -t))

    # -- chat planning ----------------------------------------------------------------------

    def _plan_message(self, snap: Snapshot) -> tuple[str, dict] | None:
        if snap.role == WOLF:
            return self._plan_wolf(snap)
        if snap.role == SEER:
            return self._plan_seer(snap)
        return self._plan_villager(snap)

    def _accusers_of(self, snap: Snapshot, seat: int) -> list[int]:
        today = [spk for spk, tgt, day in self.chat.accusations if tgt == seat and day == snap.day]
        return sorted(set(today))

    def _vote_line(self, target: int) -> str:
        self._announced_target = target
        return f"I'm voting {_p(target)}."

    def _plan_seer(self, snap: Snapshot) -> tuple[str, dict] | None:
        living = set(snap.alive)
        found_wolf = any(v == WOLF and s in living for s, v in snap.seer_results.items())
        rival = [c for c in self.chat.seer_claims if c != snap.me and c in living]
        pressured = len(self._accusers_of(snap, snap.me)) >= 2
        should_claim = found_wolf or rival or pressured or snap.day >= self.seer_claim_day or len(living) <= 4
        target = self._village_vote(snap, snap.living_others())

        if should_claim and ("claim" not in self._said or (rival and not self._reacted_claims >= set(rival))):
            self._said |= {"claim", "opening"}
            self._claimed_seer = True
            self._reacted_claims |= set(rival)
            results = ", ".join(
                f"{_p(s)} = {'WOLF' if v == WOLF else 'villager'}" for s, v in sorted(snap.seer_results.items())
            )
            parts = [f"I am the Seer. My checks: {results}." if results else "I am the Seer."]
            for r in rival:
                parts.append(f"{_p(r)} is a fake seer, so a wolf.")
            parts.append(self._vote_line(target))
            return " ".join(parts), {"intent": "seer_claim", "vote": target, "results": results}

        if "opening" not in self._said:
            self._said.add("opening")
            if should_claim:
                return self._vote_line(target), {"intent": "vote", "vote": target}
            return (
                f"No hard info yet. Let's all vote together so we don't split. {self._vote_line(target)}",
                {"intent": "vote", "vote": target},
            )
        return self._react_common(snap, target)

    def _plan_villager(self, snap: Snapshot) -> tuple[str, dict] | None:
        beliefs = self._beliefs(snap)
        target = self._village_vote(snap, snap.living_others())
        if "opening" not in self._said:
            self._said.add("opening")
            reason = self._reason(snap, target, beliefs)
            claims = [c for c in self.chat.seer_claims if c in snap.alive]
            if len(claims) == 1 and beliefs.p_wolf.get(claims[0], 1) < 0.35:
                self._reacted_claims.add(claims[0])
                lead = f"Villager here. I believe {_p(claims[0])}'s seer claim."
            elif len(claims) >= 2:
                self._reacted_claims |= set(claims)
                lead = f"Villager here. {' and '.join(_p(c) for c in claims)} both claim seer, so one is a wolf."
            elif snap.day <= 1 and not snap.votes:
                lead = "Villager here. Seer, if you found a wolf, claim now."
            else:
                lead = "Villager here."
            return f"{lead} {self._vote_line(target)} {reason}".strip(), {"intent": "vote", "vote": target}
        return self._react_common(snap, target)

    def _plan_wolf(self, snap: Snapshot) -> tuple[str, dict] | None:
        wolves = snap.partners | {snap.me}
        living_partners = snap.partners & snap.alive
        rivals = [c for c in self.chat.seer_claims if c in snap.alive and c not in wolves]
        partner_claimed = any(p in self.chat.seer_claims for p in living_partners)
        hit = [
            (c, t)
            for c in rivals
            for t, v in self.chat.claimed_results.get(c, {}).items()
            if v == WOLF and t in wolves
        ]
        villagers = [s for s in snap.living_others() if s not in wolves]
        target = self._wolf_preferred_lynch(snap, villagers) if villagers else None

        # Counter-claim when a seer claimant outs one of us.
        if self.counter_claim == "self":
            hit = [(c, t) for c, t in hit if t == snap.me]
        if self.counter_claim and hit and not partner_claimed and not self._fake_seer:
            claimant = hit[0][0]
            self._fake_seer = True
            self._fake_results[claimant] = WOLF
            self._said.add("claim")
            self._announced_target = claimant
            outed = hit[0][1]
            defend = f" {_p(outed)} is clean." if outed != snap.me else ""
            return (
                f"I am the real Seer. I checked {_p(claimant)}: WOLF. They're faking to frame us.{defend} I'm voting {_p(claimant)}.",
                {"intent": "seer_claim", "vote": claimant},
            )
        if partner_claimed and "support" not in self._said:
            self._said.add("support")
            partner = next(p for p in living_partners if p in self.chat.seer_claims)
            vote = next(
                (t for t, v in self.chat.claimed_results.get(partner, {}).items() if v == WOLF and t in snap.alive),
                target,
            )
            if vote is not None:
                self._announced_target = vote
            return (
                f"Villager here. {_p(partner)}'s seer claim makes sense to me. I'm voting {_p(vote)}." if vote is not None
                else f"Villager here. I believe {_p(partner)}.",
                {"intent": "vote", "vote": vote},
            )
        if self._fake_seer and "claim" not in self._said:
            self._said.add("claim")
            fresh = [s for s in villagers if s not in self._fake_results]
            if fresh:
                check = self.rng.choice(fresh)
                self._fake_results[check] = VILLAGER
            results = ", ".join(f"{_p(s)} = {'WOLF' if v == WOLF else 'villager'}" for s, v in self._fake_results.items())
            vote = next((t for t, v in self._fake_results.items() if v == WOLF and t in snap.alive), target)
            self._announced_target = vote
            return f"Seer again. My checks: {results}. I'm voting {_p(vote)}.", {"intent": "seer_claim", "vote": vote}
        if "opening" not in self._said and target is not None:
            self._said.add("opening")
            reason = self._reason(snap, target, self._beliefs(snap, private=False))
            lead = "Villager here."
            if snap.day <= 1 and not self.chat.seer_claims:
                lead = "Villager here, no info. Seer should speak up."
            return f"{lead} {self._vote_line(target)} {reason}".strip(), {"intent": "vote", "vote": target}
        if target is None:
            return None
        return self._react_common(snap, target)

    def _react_common(self, snap: Snapshot, target: int) -> tuple[str, dict] | None:
        accusers = [a for a in self._accusers_of(snap, snap.me) if a not in self._reacted_accusers]
        if accusers:
            self._reacted_accusers |= set(accusers)
            who = " and ".join(_p(a) for a in accusers)
            claim = "the Seer" if (snap.role == SEER and self._claimed_seer) or self._fake_seer else "a villager"
            return (
                f"I'm {claim}. {who} pushing me without evidence is exactly what a wolf would do. {self._vote_line(target)}",
                {"intent": "defend", "vote": target},
            )
        new_claims = [c for c in self.chat.seer_claims if c in snap.alive and c not in self._reacted_claims and c != snap.me]
        if new_claims:
            self._reacted_claims |= set(new_claims)
            beliefs = self._beliefs(snap, private=snap.role != WOLF)
            c = new_claims[0]
            if snap.role != WOLF and beliefs.p_wolf.get(c, 0) < 0.35:
                return f"{_p(c)}'s seer claim is credible. {self._vote_line(target)}", {"intent": "vote", "vote": target}
            return f"Noted {_p(c)}'s claim. {self._vote_line(target)}", {"intent": "vote", "vote": target}
        if target != self._announced_target:
            return f"Changing my vote: {self._vote_line(target)}", {"intent": "vote", "vote": target}
        return None

    def _reason(self, snap: Snapshot, target: int, beliefs: Beliefs) -> str:
        lynched_villagers = {d.seat for d in snap.dead.values() if d.cause and "vote" in d.cause.lower() and d.role != WOLF}
        if any(v.voter == target and v.target in lynched_villagers for v in snap.votes):
            return "They voted out an innocent villager."
        claims = self.chat.claimed_results
        for claimant, results in claims.items():
            if results.get(target) == WOLF and beliefs.p_wolf.get(claimant, 1) < 0.5:
                return f"{_p(claimant)} found them as a wolf."
        victims = {d.seat for d in snap.dead.values() if d.cause and "night" in d.cause.lower()}
        if any(spk in victims and tgt == target for spk, tgt, _ in self.chat.accusations):
            return "The wolves killed the person who suspected them."
        if target in self.chat.seer_claims:
            return "Their seer claim doesn't add up."
        if any(spk == target and tgt in self.chat.seer_claims for spk, tgt, _ in self.chat.accusations):
            return "They attacked the seer claim."
        if self.chat.message_counts.get(target, 0) == 0 and snap.day >= 1:
            return "They've said nothing useful."
        return "Their play looks off to me."

    # -- wolf partner coordination -----------------------------------------------------------

    def _private_message(self, snap: Snapshot) -> SendMessage | None:
        if snap.role != WOLF or self._told_partner_day == snap.day:
            return None
        partners = sorted(snap.partners & snap.alive)
        if not partners:
            return None
        self._told_partner_day = snap.day
        targets = [s for s in snap.living_others() if s not in snap.partners]
        if not targets:
            return None
        kill = self._kill_ranking(snap, targets)[0]
        lynch = self._wolf_preferred_lynch(snap, targets)
        self._announced_target = lynch
        return SendMessage(
            f"Partner: let's both vote {_p(lynch)} today and kill {_p(kill)} tonight. Don't defend me openly.",
            [partners[0]],
        )


def _infer_sub_phase(state: GameState) -> str:
    labels = " ".join((a.label or "") for a in state.legal_actions).lower()
    if "kill" in labels:
        return "night_wolf"
    if "investigat" in labels:
        return "night_seer"
    return "day_vote"


def create_werewolf_agent(**kwargs: Any) -> WerewolfAgent:
    return WerewolfAgent(**kwargs)
