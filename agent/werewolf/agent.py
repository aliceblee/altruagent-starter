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
import re
from typing import Any, Callable

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState, SendMessage

from .beliefs import Beliefs, infer
from .parsing import ABSTAIN, SEER, UNISON_GAP, VILLAGER, WOLF, ChatFacts, Resolver, Snapshot, read_message, snapshot

MAX_CHATS_PER_DAY = 4  # the server allows 5; keep one in reserve
MAX_WORDS = 45  # the server's cap is 50
MIN_SECONDS_TO_SPEAK = 6.0
BANDWAGON_WEIGHT = 0.06  # per announced vote, when picking among near-equal suspects
# Once we've announced a vote, keep it unless another suspect is clearly likelier. Without
# hard info the odds wobble by a few points with every message; following them made our
# agents flip-flop in chat and then vote for someone they never announced.
STICKY_MARGIN = 0.12
# Day-1 lines are the same for every role, and each comes in a few phrasings picked at random:
# other agents may keep notes across games and match our fixed sentences to the roles revealed
# at death, so no sentence may belong to one role, and no exact sentence should mark our seat.
# The opener is our first message of day 1. (The old "vegetarian vampire Eddie Cullen" joke got
# us targeted at LLM tables.) None of them mentions the seer: asking the seer to speak up on
# day 1 just points the wolves at it.
DAY1_OPENERS = (
    "Hello! No hard info yet. Anyone have a lead?",
    "Hi all. Nothing solid to go on yet. Does anyone have a lead?",
    "Morning, everyone. No real info so far. Any leads?",
    "Hey all, nothing concrete yet. Who has a lead?",
)
# Day 1 with nothing to go on: ask the table instead of naming anyone.
DAY1_ASKS = (
    "No leads yet. Who's your top suspect, and why? I'll back the strongest case so we don't split.",
    "Still nothing solid. Name your top suspect and your reason; I'll go with the best-argued case so we don't split.",
    "No real info yet. Who worries you most, and why? Let's settle on the strongest case together rather than scatter.",
)
# Following someone else's vote ({who} pushed {target}). Each says "join"/"back" so it reads as following.
JOIN_LINES = (
    "Okay, joining {who}'s vote on {target} so we don't split.",
    "Joining {who} on {target}, so we don't split. I'm voting {target}.",
    "Backing {who}'s case on {target}; one target beats a scattered vote. I'm voting {target}.",
)


def _p(seat: int) -> str:
    return f"Player{seat}"


class WerewolfAgent:
    # Strategy switches (also used by the simulator for ablations).
    counter_claim = "self"  # True: when either wolf is outed; "self": only when I am; False: never
    seer_claim_day = 2  # claim on this day at the latest, even without a wolf found
    # Never reveal ourselves as the seer on day 1 (team decision). Simulator cost, 6000 games:
    # seer win rate vs heuristic tables 56.8% -> 48.2%, self-play village 51.8% -> 31.2%,
    # because a wolf found on night 1 is no longer claimed and voted out on day 1.
    seer_hides_day1 = True
    rescue = True  # rally the table onto our top suspect when votes pile up on us
    call_out_unison = True  # villager/seer: point out two players pushing the same target in unison
    # Wolf play.
    # "accuser": "I checked my accuser: WOLF"; "other": a check on someone else plus "my accuser is
    # lying". "other" avoids the reactive-claim tell but tested no better against smart villagers
    # (43% vs 43%) and worse against simple ones (74% vs 85%), so it's off by default.
    counter_claim_style = "accuser"
    spare_accusers = True  # don't kill players who accused us (their death points at us)
    endgame_ties = True  # with 5 or fewer alive, vote to steer the tally away from a wolf
    split_votes = True  # when a lynch is already decided, don't add our vote next to our partner's
    # Day 1: voice mild suspicion of our partner. Off: our villagers never name anyone on day 1
    # without a reason, so the fixed aside would point straight at the other wolf.
    distance_partner = False

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
        self._session: str = ""
        self._voice = (rng or random.Random()).choice(_VOICES)
        self._resolver: Resolver | None = None
        self._reacted_claims: set[int] = set()
        self._reacted_accusers: set[int] = set()
        self._announced_target: int | None = None
        self._answered_asks = 0  # how many of chat.asked_me we've already answered
        # Seer state.
        self._claimed_seer = False
        # Wolf state.
        self._fake_seer = False
        self._fake_results: dict[int, str] = {}
        self._told_partner_day: int | None = None
        self._planned_kill: tuple[int, int] | None = None  # (day, seat) we told our partner
        self._fake_accused: set[int] = set()  # who we called a liar / wolf as fake seer
        self._claim_busted = False  # our fake seer claim was disproved by a revealed role
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
            text = self.rng.choice(DAY1_OPENERS)
            self.log(f"[werewolf] P{snap.me} says: {text}")
            return SendMessage(text)

        private = self._private_message(snap)
        if private is not None:
            self._sent_today += 1
            return private

        plan = self._plan_message(snap)
        if plan is None:
            self._done_today = True
            return TERMINATE_MESSAGING
        text, facts = plan
        if self.polish is not None and not facts.get("verbatim"):
            try:
                text = self.polish(text, {**facts, "voice": self._voice}) or text
            except Exception as exc:  # noqa: BLE001 - never let chat polish break the game
                self.log(f"[werewolf] chat polish failed: {exc!r}")
        text = self._with_readable_vote(text, facts.get("vote"))
        self._sent_today += 1
        self.log(f"[werewolf] P{snap.me} says: {text}")
        return SendMessage(text)

    def _with_readable_vote(self, text: str, vote: Any) -> str:
        """Keep the vote machine-readable after an LLM rewrite ("I'm convinced Player0
        is a wolf" reads as an accusation, not a vote), so other agents can join it."""
        words = text.split()
        if not isinstance(vote, int) or self._resolver is None:
            return " ".join(words[:MAX_WORDS])
        probe = ChatFacts()
        read_message(probe, -1, text, [], 0, -2, self._resolver)
        if probe.intents_on(0).get(-1) == vote or (-1, vote, 0) in probe.accusations:
            return " ".join(words[:MAX_WORDS])  # already reads as pushing that player
        tail = f"I'm voting {_p(vote)}."
        return " ".join(words[: MAX_WORDS - len(tail.split())] + tail.split())

    # -- state ------------------------------------------------------------------------------

    def _update(self, state: GameState, context: DecisionContext) -> tuple[Snapshot, Resolver]:
        snap, resolver = snapshot(state, context.seat_position)
        self._session = context.session_id
        self._resolver = resolver
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
        # A partner's private suggestion wins (on a disagreement the server takes
        # the lower seat's target anyway); else the target we suggested to them.
        suggestion = self._partner_suggestion(snap, targets)
        if suggestion is not None:
            return suggestion
        if self._planned_kill and self._planned_kill[1] in targets and self._planned_kill[0] >= snap.day - 1:
            return self._planned_kill[1]
        return self._kill_ranking(snap, targets)[0]

    def _kill_ranking(self, snap: Snapshot, targets: list[int]) -> list[int]:
        public = self._beliefs(snap, private=False)
        wolves = snap.partners | {snap.me}

        def score(t: int) -> float:
            s = 0.0
            if t in self.chat.seer_claims:
                s += 5.0  # a live seer claimant is the biggest threat
            s += 3.0 * public.p_seer.get(t, 0.0)
            pressure = sum(1 for spk, tgt, _ in self.chat.accusations if spk == t and tgt in wolves)
            pressure += sum(1 for v in snap.votes if v.voter == t and v.target in wolves)
            if self.spare_accusers:
                # Killing whoever was onto us is the classic tell ("the victim suspected X").
                # Leave them alive: their accusation then looks like a grudge.
                s -= 0.7 * pressure
            else:
                s += 0.7 * pressure
            # Players the village trusts carry weight in discussion; a cleared player
            # is also a confirmed villager, which makes them a strong voice.
            s += 1.0 * (1.0 - public.p_wolf.get(t, 0.0))
            s += 0.5 * sum(1 for _, tgt, _ in self.chat.clears if tgt == t)
            return s + self.rng.random() * 0.05

        return sorted(targets, key=score, reverse=True)

    def _partner_suggestion(self, snap: Snapshot, targets: list[int]) -> int | None:
        """The kill target named in our partner's latest private message (today or
        last night), e.g. "kill Player4 tonight", "target 4", "let's eat P4"."""
        for sender, text, day in reversed(self.chat.private_from):
            if sender in snap.partners and day >= snap.day - 1:
                seat = _named_after(text, _KILL_WORDS, targets)
                if seat is not None:
                    return seat
        return None

    def _partner_vote_suggestion(self, snap: Snapshot, targets: list[int]) -> int | None:
        for sender, text, day in reversed(self.chat.private_from):
            if sender in snap.partners and day == snap.day:
                seat = _named_after(text, _VOTE_WORDS, targets)
                if seat is not None:
                    return seat
        return None

    # -- day vote ---------------------------------------------------------------------------

    def _day_vote(self, snap: Snapshot, seats: list[int]) -> int:
        if snap.role == WOLF:
            return self._wolf_vote(snap, seats)
        return self._village_vote(snap, seats)

    def _village_vote(self, snap: Snapshot, seats: list[int]) -> int:
        beliefs = self._beliefs(snap)
        spared = self._spared_claimants(snap)
        seats = [s for s in seats if s not in spared] or seats
        if snap.role == SEER:
            known_wolves = [s for s in seats if snap.seer_results.get(s) == WOLF]
            if known_wolves:
                return known_wolves[0]
        if self._announced_target in seats and self._no_hard_info(snap):
            return self._announced_target  # nothing better to go on: keep our word, don't split
        intents = self.chat.intents_on(snap.day)

        def score(t: int) -> float:
            backers = sum(1 for spk, tgt in intents.items() if tgt == t and beliefs.p_wolf.get(spk, 0) < 0.5)
            return beliefs.p_wolf.get(t, 0.0) + BANDWAGON_WEIGHT * backers

        best = max(score(s) for s in seats)
        if self._announced_target in seats and best - score(self._announced_target) < STICKY_MARGIN:
            return self._announced_target
        top = [s for s in seats if score(s) >= best - 1e-9]
        return min(top)  # deterministic, so like-minded villagers converge

    def _wolf_vote(self, snap: Snapshot, seats: list[int]) -> int:
        partners = snap.partners & snap.alive
        wolves = partners | {snap.me}
        intents = self.chat.intents_on(snap.day)
        alive = len(snap.alive)
        villagers = [s for s in seats if s not in partners]
        choice = self._wolf_normal_vote(snap, seats, partners, villagers, intents, alive)
        if self.endgame_ties and alive <= 5:
            steer = self._steer_tally(snap, seats, wolves, villagers, default=choice)
            if steer is not None:
                return steer
        return choice

    def _wolf_normal_vote(self, snap, seats, partners, villagers, intents, alive) -> int:
        for p in partners:
            against = sum(1 for spk, tgt in intents.items() if tgt == p and spk != snap.me)
            if against * 2 > alive - 1 and p in seats:
                return p  # the partner is going anyway: join in and look like a villager
        if not villagers:
            return seats[0]
        choice = self._wolf_preferred_lynch(snap, villagers)
        if self.split_votes and partners:
            choice = self._spread_vote(snap, choice, villagers)
        return choice

    def _projected_tally(self, snap: Snapshot) -> dict[int, int]:
        """Votes announced today by living players other than me (latest intent each)."""
        tally: dict[int, int] = {}
        for voter, target in self.chat.intents_on(snap.day).items():
            if voter != snap.me and voter in snap.alive and target in snap.alive:
                tally[target] = tally.get(target, 0) + 1
        return tally

    @staticmethod
    def _outcome(tally: dict[int, int]) -> int | None:
        """Who a plurality vote eliminates (None on a tie or no votes)."""
        if not tally:
            return None
        top = max(tally.values())
        leaders = [t for t, c in tally.items() if c == top]
        return leaders[0] if len(leaders) == 1 else None

    def _steer_tally(self, snap: Snapshot, seats: list[int], wolves: set[int], villagers: list[int],
                     default: int | None = None) -> int | None:
        """Endgame: with the announced votes in hand, cast ours so that a villager
        goes out, else nobody (a tie, then a night kill, reaches parity), and
        never a wolf. Returns None when the announcements don't decide it."""
        base = self._projected_tally(snap)
        if not base:
            return None

        def rank(choice: int) -> tuple[int, int, int]:
            tally = dict(base)
            if choice != ABSTAIN:
                tally[choice] = tally.get(choice, 0) + 1
            out = self._outcome(tally)
            value = 2 if out is not None and out not in wolves else (1 if out is None else 0)
            # Equal outcomes: add our vote to the leader (a safer margin), else any villager.
            return value, int(choice == out), int(choice in villagers)

        options = [s for s in seats if s not in wolves] + [ABSTAIN]
        best = max(options, key=rank)
        # Only override our normal vote when it actually changes the result (in testing it
        # picked an arbitrary villager, contradicting what we'd announced, when it couldn't).
        if default is not None and rank(best)[0] <= rank(default)[0]:
            return None
        return best  # may be ABSTAIN, when that's what keeps the tally tied

    def _spread_vote(self, snap: Snapshot, choice: int, villagers: list[int]) -> int:
        """If the announced votes already decide today's lynch without us, vote for
        someone else so the vote history doesn't show both wolves on the same target."""
        base = self._projected_tally(snap)
        decided = self._outcome(base)
        if decided != choice:
            return choice
        others = sorted(base.values(), reverse=True)
        lead = base[choice] - (others[1] if len(others) > 1 else 0)
        if lead < 2 or base[choice] * 2 <= len(snap.alive) - 1:
            return choice  # not safely decided: our vote may matter
        alternatives = [v for v in villagers if v != choice]
        if not alternatives:
            return choice
        public = self._beliefs(snap, private=False)
        return max(alternatives, key=lambda t: (public.p_wolf.get(t, 0.0), -t))

    def _wolf_preferred_lynch(self, snap: Snapshot, villagers: list[int]) -> int:
        public = self._beliefs(snap, private=False)
        intents = self.chat.intents_on(snap.day)

        partner_pick = self._partner_vote_suggestion(snap, villagers)

        def score(t: int) -> float:
            backers = sum(1 for spk, tgt in intents.items() if tgt == t and spk != snap.me)
            s = public.p_wolf.get(t, 0.0) + 0.15 * backers
            if t in self.chat.seer_claims:
                s += 0.6  # a real seer is the best possible lynch
            if self._fake_seer and (self._fake_results.get(t) == WOLF or t in self._fake_accused):
                s += 1.0  # stay consistent with our fake claim
            if t == partner_pick:
                s += 0.3
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
        if snap.day <= 1 and self.seer_hides_day1:
            # Never reveal ourselves on day 1, whatever happens: a day-1 seer is the night's kill.
            should_claim = False
            line = self._seer_day1(snap, rival, found_wolf)
            if line is not None:
                return line

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
            if self._uses_day1_lines(snap):
                line = self._day1_line(snap)
                if line is not None:
                    return line
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
            else:
                lead = "Villager here."
                if self._no_hard_info(snap):
                    line = self._day1_line(snap)
                    if line is not None:
                        return line
            return f"{lead} {self._vote_line(target)} {reason}".strip(), {"intent": "vote", "vote": target}
        return self._react_common(snap, target)

    def _plan_wolf(self, snap: Snapshot) -> tuple[str, dict] | None:
        self._update_fake_claim(snap)
        wolves = snap.partners | {snap.me}
        living_partners = snap.partners & snap.alive
        rivals = [c for c in self.chat.seer_claims if c in snap.alive and c not in wolves]
        partner_claimed = any(
            p in self.chat.seer_claims and not self._claim_disproved(snap, p, self.chat.claimed_results.get(p, {}))
            for p in living_partners
        )
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
        if self.counter_claim and hit and not partner_claimed and not self._fake_seer and not self._claim_busted:
            claimant = hit[0][0]
            self._fake_seer = True
            self._said |= {"claim", "opening", "support"}  # never follow it with "Villager here"
            self._announced_target = claimant
            outed = hit[0][1]
            defend = f" {_p(outed)} is clean." if outed != snap.me else ""
            if self.counter_claim_style == "accuser":
                self._fake_results[claimant] = WOLF
                return (
                    f"I am the real Seer. I checked {_p(claimant)}: WOLF. They're faking to frame us.{defend} I'm voting {_p(claimant)}.",
                    {"intent": "seer_claim", "vote": claimant},
                )
            # "I checked my accuser and they're a wolf" is the giveaway a smart table
            # looks for, so report a check on somebody else and just call them a liar.
            self._fake_accused.add(claimant)
            fresh = [s for s in villagers if s != claimant] or villagers
            check = self.rng.choice(fresh) if fresh else None
            result = ""
            if check is not None:
                self._fake_results[check] = VILLAGER
                result = f" My check was {_p(check)}: villager."
            return (
                f"I'm the real Seer.{result} {_p(claimant)} is lying about me, and only a wolf would fake that.{defend} I'm voting {_p(claimant)}.",
                {"intent": "seer_claim", "vote": claimant},
            )
        if partner_claimed and "support" not in self._said:
            self._said.add("support")
            partner = next(p for p in living_partners if p in self.chat.seer_claims
                           and not self._claim_disproved(snap, p, self.chat.claimed_results.get(p, {})))
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
            self._said |= {"claim", "opening", "support"}
            fresh = [s for s in villagers if s not in self._fake_results and s not in self._fake_accused]
            if fresh:
                # Each new "check": find the villager the table already doubts to be a wolf.
                public = self._beliefs(snap, private=False)
                check = max(fresh, key=lambda t: (public.p_wolf.get(t, 0.0), -t))
                self._fake_results[check] = WOLF
            vote = next(
                (t for t in list(self._fake_accused) + [t for t, v in self._fake_results.items() if v == WOLF] if t in snap.alive),
                None,
            )
            if vote is None and target is not None:
                self._fake_results[target] = WOLF  # the claim and the vote must agree
                vote = target
            if vote is None:
                return None
            results = ", ".join(f"{_p(s)} = {'WOLF' if v == WOLF else 'villager'}" for s, v in self._fake_results.items())
            self._announced_target = vote
            return f"I'm the Seer. My checks: {results}. I'm voting {_p(vote)}.", {"intent": "seer_claim", "vote": vote}
        if "opening" not in self._said and target is not None and self._claim_busted:
            self._said.add("opening")
            reason = self._reason(snap, target, self._beliefs(snap, private=False))
            return f"{self._vote_line(target)} {reason}".strip(), {"intent": "vote", "vote": target}
        if "opening" not in self._said and target is not None:
            self._said.add("opening")
            reason = self._reason(snap, target, self._beliefs(snap, private=False))
            aside = self._partner_aside(snap)
            if self._no_info(snap, self._beliefs(snap, private=False)):
                ask = self._day1_ask() if self._no_hard_info(snap) else None
                if ask is not None:  # day 1: the same words a villager would use
                    return ask[0] + aside, ask[1]
                return f"Villager here. No info yet, so no vote from me until the Seer speaks.{aside}", {"intent": "wait"}
            follow = self._follow_line(snap, target)
            if follow is not None:
                return f"Villager here. {follow}{aside}".strip(), {"intent": "vote", "vote": target}
            return f"Villager here. {self._vote_line(target)} {reason}{aside}".strip(), {"intent": "vote", "vote": target}
        if target is None:
            return None
        return self._react_common(snap, target)

    def _claim_disproved(self, snap: Snapshot, claimant: int, results: dict[int, str]) -> bool:
        """A seer claim the revealed roles have proved false: another player died as
        the seer, or a "wolf" result died a non-wolf (or a "villager" result a wolf)."""
        for seat, death in snap.dead.items():
            if death.role == SEER and seat != claimant:
                return True
            said = results.get(seat)
            if said and death.role and (said == WOLF) != (death.role == WOLF):
                return True
        return False

    def _update_fake_claim(self, snap: Snapshot) -> None:
        if self._fake_seer and self._claim_disproved(snap, snap.me, {**self._fake_results, **{t: WOLF for t in self._fake_accused}}):
            # Proven a liar: stop the act. Repeating the busted results only reminds the table.
            self._fake_seer = False
            self._claim_busted = True
            self.log(f"[werewolf] P{snap.me}: fake seer claim disproved, dropping it")

    def _seer_day1(self, snap: Snapshot, rival: list[int], found_wolf: bool) -> tuple[str, dict] | None:
        """Day 1 as the seer, staying hidden: we talk exactly like a villager (same lines), and
        a wolf we found is never named in chat; we vote it, and join only a push on it."""
        if "opening" in self._said:
            return None
        self._said.add("opening")
        return self._day1_line(snap)

    def _uses_day1_lines(self, snap: Snapshot) -> bool:
        """Villagers, and the seer (it never claims on day 1), on day 1."""
        if not self._no_hard_info(snap):
            return False
        if snap.role == SEER:
            return not self._claimed_seer
        return snap.role in (VILLAGER, None)

    def _protected(self, snap: Snapshot) -> set[int]:
        """Players a non-wolf must never call a wolf: a seat our seer checked as innocent,
        and a credible seer claimant (in testing, our agents attacked the real seer for
        a misread remark and other agents rightly turned on us for it)."""
        if snap.role == WOLF:
            return set()
        protected = {t for t, v in snap.seer_results.items() if v != WOLF}
        beliefs = self._beliefs(snap)
        protected |= {c for c in self.chat.seer_claims
                      if c in snap.alive and c != snap.me and beliefs.p_wolf.get(c, 1.0) < 0.35}
        return protected | self._spared_claimants(snap)

    def _spared_claimants(self, snap: Snapshot) -> set[int]:
        """A lone seer claimant from day 1, whom the village side never votes for or calls a
        wolf (team rule: a lone day-1 claim is usually the real seer; fakes tend to come as
        counterclaims).
        Not when two claimed on day 1 (one is lying), when the revealed roles disprove the
        claim, or when we are the seer (then the claim is certainly fake)."""
        if snap.role in (WOLF, SEER):
            return set()
        day1 = [c for c, d in self.chat.seer_claims.items() if d is not None and d <= 1 and c != snap.me]
        if len(day1) != 1 or day1[0] not in snap.alive:
            return set()
        claimant = day1[0]
        if self._claim_disproved(snap, claimant, self.chat.claimed_results.get(claimant, {})):
            return set()
        return {claimant}

    def _no_hard_info(self, snap: Snapshot) -> bool:
        """Day 1 before anything verifiable: no seer claim, nobody dead, no votes cast.
        (Announced votes and accusations are talk, not information.)"""
        return snap.day <= 1 and not self.chat.seer_claims and not snap.dead and not snap.votes

    def _day1_line(self, snap: Snapshot) -> tuple[str, dict] | None:
        """A villager (or the hidden seer) on day 1 with nothing to go on: fight back against a
        vote on us, else join the leading announced vote, else ask the table for reads (once).
        We never name a random suspect: other agents ask for the reason, and a made-up one gets
        us caught. A seer that found a wolf only fights back against, or joins a push on, that
        wolf (it votes the wolf whatever it says). None when there's nothing new to say."""
        if snap.day > 1:
            return None
        # Announced votes, plus plain accusations ("I'm convinced Player0 is a wolf"): an
        # LLM-written message often pushes a target without the word "vote". Latest per speaker.
        intents: dict[int, int] = {}
        for speaker, t, day in self.chat.accusations:
            if day == snap.day and speaker != snap.me and speaker in snap.alive and t in snap.alive:
                intents.pop(speaker, None)
                intents[speaker] = t
        cleared = {t for t, v in snap.seer_results.items() if v != WOLF} | self._spared_claimants(snap)
        known = [t for t, v in snap.seer_results.items() if v == WOLF and t in snap.alive]
        allowed = set(known) if known else None  # a seer with a wolf only ever names that wolf
        against_me = [voter for voter, t in intents.items()
                      if t == snap.me and voter not in self._reacted_accusers and voter not in self._protected(snap)
                      and (allowed is None or voter in allowed)]
        if against_me and "day1_accuse" not in self._said:
            # Once: name the first accuser; if more votes pile on, the rescue message follows.
            self._said.add("day1_accuse")
            accuser = against_me[0]
            self._reacted_accusers |= set(against_me)
            self._announced_target = accuser
            return (f"{_p(accuser)}, you're being really suspicious, quick to point fingers already. "
                    "I think you're the werewolf.", {"intent": "vote", "vote": accuser})
        if self._announced_target is not None:
            return None
        backers: dict[int, list[int]] = {}
        for voter, t in intents.items():  # in the order they were announced
            if t != snap.me and t not in cleared and (allowed is None or t in allowed):
                backers.setdefault(t, []).append(voter)
        if backers:
            pick = max(backers, key=lambda t: len(backers[t]))  # most backed; first announced on a tie
            self._announced_target = pick
            return self._join_line(backers[pick][0], pick), {"intent": "vote", "vote": pick}
        return self._day1_ask()

    def _day1_ask(self) -> tuple[str, dict] | None:
        """Ask the table for reads, once a day 1."""
        if "day1_ask" in self._said:
            return None
        self._said.add("day1_ask")
        return self.rng.choice(DAY1_ASKS), {"intent": "ask"}

    def _join_line(self, who: int, target: int) -> str:
        return self.rng.choice(JOIN_LINES).format(who=_p(who), target=_p(target))

    def _follow_line(self, snap: Snapshot, target: int) -> str | None:
        """Wolf: if our partner just pushed the same target, echoing it right away is the
        "two players in unison" tell smart villagers look for. Present our vote as
        following someone instead, crediting a non-partner who pushed it if there is one."""
        recent = [(s, n) for s, t, d, n in self.chat.pushes
                  if d == snap.day and t == target and self.chat.public_messages - n <= UNISON_GAP + 1]
        if not any(s in snap.partners for s, _ in recent):
            return None
        backers = [s for s, t, d, _ in self.chat.pushes if d == snap.day and t == target and s != snap.me]
        credit = next((s for s in backers if s not in snap.partners), backers[0])
        self._announced_target = target
        return self._join_line(credit, target)

    def _partner_aside(self, snap: Snapshot) -> str:
        """Day 1 only: a mild word of doubt about our partner, so the two of us don't
        read as a pair. Never when the partner is already under real suspicion."""
        living = sorted(snap.partners & snap.alive)
        if not self.distance_partner or snap.day > 1 or not living:
            return ""
        partner = living[0]
        if self._accusers_of(snap, partner) or any(r.get(partner) == WOLF for r in self.chat.claimed_results.values()):
            return ""
        return f" {_p(partner)} seems a little suspicious to me, though."

    def _no_info(self, snap: Snapshot, beliefs: Beliefs) -> bool:
        """Nothing distinguishes the other living players yet (typically day 1 before
        any claim): naming a target now would only be a coin flip that splits the table."""
        values = [beliefs.p_wolf.get(s, 0.0) for s in snap.living_others()]
        return not values or max(values) - min(values) < 0.02

    def _unison_call_out(self, snap: Snapshot, target: int) -> tuple[str, dict] | None:
        """Villager or seer: two other players jumped on the same target at the same time,
        unprompted. That's how a wolf pair coordinates, so say so (once per pair a day)."""
        if not self.call_out_unison or snap.role == WOLF:
            return None
        for first, second, pushed, day in self.chat.unison:
            key = "unison"  # once a day: three agents posting eight call-outs in testing was spam
            if day != snap.day or key in self._said or snap.me in (first, second):
                continue
            if first not in snap.alive or second not in snap.alive:
                continue
            self._said.add(key)
            who = "me" if pushed == snap.me else _p(pushed)
            return (
                f"{_p(first)} looks suspicious, and {_p(second)} looks suspicious too: they jumped on {who} "
                f"in unison, with nothing to go on. That's how two werewolves coordinate. I'm watching them both.",
                {"intent": "suspect", "vote": None},
            )
        return None

    def _votes_against_me(self, snap: Snapshot) -> list[int]:
        """Living players whose latest announced vote today is me."""
        return sorted(s for s, t in self.chat.intents_on(snap.day).items() if t == snap.me and s in snap.alive)

    def _rescue(self, snap: Snapshot, target: int) -> tuple[str, dict] | None:
        """Once a day, when announced votes against us approach a majority of the other
        voters, spend a message pulling the table onto our top suspect instead."""
        if not self.rescue or "rescue" in self._said or target == snap.me:
            return None
        against = self._votes_against_me(snap)
        voters = len(snap.alive) - 1  # everyone but me
        # At least 2 votes, and within one vote of half the other voters
        # (7 alive: 3 votes; 5 alive: 2; 4 alive: 2).
        if len(against) < 2 or len(against) * 2 < voters - 1:
            return None
        self._said.add("rescue")
        self._reacted_accusers |= set(against)
        beliefs = self._beliefs(snap, private=snap.role != WOLF)
        claim = "the Seer" if (snap.role == SEER and self._claimed_seer) or self._fake_seer else "a villager"
        who_i_am = "" if self._claim_busted else f", and I'm {claim}"
        reason = self._reason(snap, target, beliefs)
        return (
            f"Stop: {len(against)} votes on me{who_i_am}. A wrong lynch hands the wolves the game. "
            f"The real case is {_p(target)}: {reason} Everyone, vote {_p(target)} with me.",
            {"intent": "rescue", "vote": self._mark_announced(target)},
        )

    def _mark_announced(self, target: int) -> int:
        self._announced_target = target
        return target

    def _react_common(self, snap: Snapshot, target: int) -> tuple[str, dict] | None:
        if self._uses_day1_lines(snap):
            line = self._day1_line(snap)
            if line is not None:
                return line
        rescue = self._rescue(snap, target)
        if rescue is not None:
            return rescue
        if snap.role == SEER and self._uses_day1_lines(snap):
            return None  # the hidden seer says no more than a villager would on day 1, and just votes
        accusers = [a for a in self._accusers_of(snap, snap.me) if a not in self._reacted_accusers]
        protected = self._protected(snap)
        friendly = [a for a in accusers if a in protected and a in self.chat.seer_claims]
        accusers = [a for a in accusers if a not in protected]
        if friendly and not accusers:
            self._reacted_accusers |= set(friendly)
            return (f"{_p(friendly[0])}, I'm on your side, not against you. {self._vote_line(target)}",
                    {"intent": "vote", "vote": target})
        if accusers:
            self._reacted_accusers |= set(accusers)
            who = " and ".join(_p(a) for a in accusers)
            claim = "the Seer" if (snap.role == SEER and self._claimed_seer) or self._fake_seer else "a villager"
            self._answered_asks = len(self.chat.asked_me)  # the read below answers them too
            read = self._discreet_read(snap, target, brief=True)
            if self._claim_busted:
                return (f"{who} pushing me is exactly what a wolf would do. {read} {self._vote_line(target)}",
                        {"intent": "defend", "vote": target})
            return (
                f"I'm {claim}. {who} pushing me without evidence is what a wolf would do. {read} {self._vote_line(target)}",
                {"intent": "defend", "vote": target},
            )
        askers = [a for a, d in self.chat.asked_me[self._answered_asks:] if d == snap.day and a in snap.alive]
        self._answered_asks = len(self.chat.asked_me)
        if askers and not self._uses_day1_lines(snap):
            read = self._discreet_read(snap, target)
            return (f"{_p(askers[-1])}, {read[0].lower()}{read[1:]} {self._vote_line(target)}",
                    {"intent": "vote", "vote": target})
        unison = self._unison_call_out(snap, target)
        if unison is not None:
            return unison
        new_claims = [c for c in self.chat.seer_claims if c in snap.alive and c not in self._reacted_claims and c != snap.me]
        if new_claims:
            self._reacted_claims |= set(new_claims)
            beliefs = self._beliefs(snap, private=snap.role != WOLF)
            c = new_claims[0]
            if snap.role != WOLF and beliefs.p_wolf.get(c, 0) < 0.35:
                return f"{_p(c)}'s seer claim is credible. {self._vote_line(target)}", {"intent": "vote", "vote": target}
            return f"Noted {_p(c)}'s claim. {self._vote_line(target)}", {"intent": "vote", "vote": target}
        if target != self._announced_target:
            if self._announced_target is None:
                if self._no_info(snap, self._beliefs(snap, private=snap.role != WOLF)) and not self.chat.intents_on(snap.day):
                    return None  # still nothing to go on; the vote itself will follow the table
                return f"{self._vote_line(target)} {self._reason(snap, target, self._beliefs(snap, private=snap.role != WOLF))}", {"intent": "vote", "vote": target}
            return f"Changing my vote: {self._vote_line(target)}", {"intent": "vote", "vote": target}
        return None

    def _discreet_read(self, snap: Snapshot, target: int, *, brief: bool = False) -> str:
        """Our current scoring in words, never numbers: the vote target, a second suspect,
        and someone we trust. A wolf ranks from public information and never names its
        partner either way. ``brief``: the target only (when defending ourselves; naming
        two more players there cost ~2 points of village wins in self-play)."""
        wolf = snap.role == WOLF
        beliefs = self._beliefs(snap, private=not wolf)
        pool = [s for s in snap.living_others() if not (wolf and s in snap.partners)]
        protected = self._protected(snap)
        ranked = sorted((s for s in pool if s != target and s not in protected),
                        key=lambda s: (-beliefs.p_wolf.get(s, 0.0), s))
        p = beliefs.p_wolf.get(target, 0.0)
        lead = (f"I'm fairly sure about {_p(target)}" if p >= 0.6 else f"{_p(target)} worries me most"
                if p >= 0.4 else f"I lean slightly toward {_p(target)}")
        if brief:
            return f"My read so far: {lead}."
        second = f", {_p(ranked[0])} next" if ranked and beliefs.p_wolf.get(ranked[0], 0.0) >= 0.3 else ""
        trusted = [s for s in sorted(pool, key=lambda s: (beliefs.p_wolf.get(s, 0.0), s))
                   if s != target and (not ranked or s != ranked[0])]
        clean = f"; {_p(trusted[0])} reads clean to me" if trusted and beliefs.p_wolf.get(trusted[0], 1.0) < 0.25 else ""
        return f"My read so far: {lead}{second}{clean}."

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
        credible = {c for c in self.chat.seer_claims
                    if c != target and beliefs.p_wolf.get(c, 1.0) < 0.35 and snap.dead.get(c, None) is None}
        credible |= {s for s, d in snap.dead.items() if d.role == SEER and s in self.chat.seer_claims}
        if any(spk == target and tgt in credible for spk, tgt, _ in self.chat.accusations):
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
        if snap.day >= 2 and not any(sender in snap.partners for sender, _, _ in self.chat.private_from):
            return None  # the partner has never answered: save the message
        targets = [s for s in snap.living_others() if s not in snap.partners]
        if not targets:
            return None
        # Vote target first, then a different kill target: if today's vote goes our
        # way, that player is already gone by night, so a shared target wastes the kill.
        lynch = self._wolf_preferred_lynch(snap, targets)
        self._announced_target = lynch
        kill_targets = [t for t in targets if t != lynch] or targets
        partner_kill = self._partner_suggestion(snap, kill_targets)
        kill = partner_kill if partner_kill is not None else self._kill_ranking(snap, kill_targets)[0]
        self._planned_kill = (snap.day, kill)
        template = self.rng.choice(_WHISPERS)
        return SendMessage(template.format(lynch=_p(lynch), kill=_p(kill)), [partners[0]])


# Each agent rewrites its messages in its own voice, so two of ours at one table don't post
# near-identical text (other agents flagged that as "copy-paste coordination").
_VOICES = ("blunt and brief", "calm and analytical", "warm and conversational", "dry and witty", "earnest and direct")

_WHISPERS = (
    "Partner: vote {lynch} today, kill {kill} tonight. Don't defend me openly.",
    "Wolf plan: we push {lynch} today; tonight we kill {kill}. Keep your distance from me in public.",
    "Let's both go for {lynch} at the vote and kill {kill} tonight. I may doubt you in public, ignore it.",
)
_KILL_WORDS = re.compile(r"\b(?:kill|killing|target|eat|bite|tonight|night)\b", re.I)
_VOTE_WORDS = re.compile(r"\b(?:vote|voting|lynch|push|eliminate)\b", re.I)
_SEAT_AFTER = re.compile(r"^\W{0,3}(?:(?:player|p|seat)\s*)?(\d)\b", re.I)


def _named_after(text: str, keywords: re.Pattern, targets: list[int]) -> int | None:
    """The first target seat named right after one of ``keywords`` ("kill Player4",
    "target 4", "eat P4"), or right before "tonight" ("Player4 tonight")."""
    for m in keywords.finditer(text):
        after = text[m.end(): m.end() + 20]
        found = _SEAT_AFTER.match(after)
        if found and int(found.group(1)) in targets:
            return int(found.group(1))
    for m in re.finditer(r"(?:player|p)\s*(\d)\s+(?:tonight|at\s+night)", text, re.I):
        if int(m.group(1)) in targets:
            return int(m.group(1))
    return None


def _infer_sub_phase(state: GameState) -> str:
    labels = " ".join((a.label or "") for a in state.legal_actions).lower()
    if "kill" in labels:
        return "night_wolf"
    if "investigat" in labels:
        return "night_seer"
    return "day_vote"


def create_werewolf_agent(**kwargs: Any) -> WerewolfAgent:
    return WerewolfAgent(**kwargs)
