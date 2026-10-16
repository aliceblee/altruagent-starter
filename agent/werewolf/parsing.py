"""Turn GameAPI's Werewolf payloads and the day's chat into plain facts.

The platform's exact private-observation wording isn't documented in this
starter, so everything here is deliberately tolerant: structured fields are
preferred when present (``state.raw["game_state"]``, ``players``, any
role/teammate/investigation keys), and the free-text observation is read with
forgiving patterns. Anything that can't be read is simply left unknown; the
belief model works with whatever it gets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

ABSTAIN = 7

WOLF, SEER, VILLAGER = "wolf", "seer", "villager"

_PLAYER_RE = re.compile(r"\bplayer\s*[_#-]?\s*(\d+)\b", re.I)
# Other agents often abbreviate: "P6's counter-claim", "P1 and P2". Capital P only.
_SHORT_RE = re.compile(r"\bP\s?(\d)\b")
_SENTENCE_SPLIT = re.compile(r"[.!?\n;]+")


def normalize_role(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    if not t:
        return None
    if re.search(r"\bnot\s+(?:a\s+|the\s+)?(?:were)?wolf\b", t):
        return VILLAGER
    if re.search(r"(?:were)?wol(?:f|ves)\b|\bevil\b|\bbad\b", t):
        return WOLF
    if "seer" in t:
        return SEER
    if re.search(r"villager|village|innocent|\bgood\b|\bclean\b|\bcleared?\b|\btown\b", t):
        return VILLAGER
    return None


@dataclass
class Death:
    seat: int
    role: str | None
    cause: str | None  # "night_kill" / "vote" / "resigned" / None
    day: int | None


@dataclass
class Vote:
    day: int | None
    voter: int
    target: int | None  # None = abstain / no counted target


@dataclass
class Snapshot:
    """Everything this seat knows from the server right now."""

    me: int
    n_players: int
    names: dict[int, str]
    alive: set[int]
    dead: dict[int, Death]
    role: str | None
    partners: set[int]
    seer_results: dict[int, str]  # target -> WOLF / VILLAGER (only when I'm the seer)
    day: int
    sub_phase: str | None
    votes: list[Vote]
    seconds_left: float | None = None

    def living_others(self) -> list[int]:
        return sorted(s for s in self.alive if s != self.me)


class Resolver:
    """Map display names / ``PlayerN`` / ints to seat positions."""

    def __init__(self, players: Iterable[dict], n_players: int) -> None:
        self.n = n_players
        self.by_name: dict[str, int] = {}
        for p in players:
            pos, name = p.get("position"), p.get("name")
            if isinstance(pos, int) and isinstance(name, str) and name.strip():
                self.by_name[name.strip().casefold()] = pos
        # Longest names first, so "Player 12" isn't matched as "Player 1".
        self._name_patterns = [
            (re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I), pos)
            for name, pos in sorted(self.by_name.items(), key=lambda kv: -len(kv[0]))
            if not _PLAYER_RE.fullmatch(name)
        ]

    def seat(self, value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value if 0 <= value < self.n else None
        if isinstance(value, dict):
            for key in ("position", "seat", "player", "name", "id"):
                if key in value:
                    return self.seat(value[key])
            return None
        if isinstance(value, str):
            text = value.strip()
            if text.casefold() in self.by_name:
                return self.by_name[text.casefold()]
            if text.isdigit():
                return self.seat(int(text))
            match = _PLAYER_RE.fullmatch(text) or _SHORT_RE.fullmatch(text)
            if match:
                return self.seat(int(match.group(1)))
        return None

    def mentions(self, text: str) -> list[tuple[int, int, int]]:
        """(seat, start, end) for every player mentioned in ``text``, in order."""
        found: list[tuple[int, int, int]] = []
        for m in list(_PLAYER_RE.finditer(text)) + list(_SHORT_RE.finditer(text)):
            seat = int(m.group(1))
            if 0 <= seat < self.n:
                found.append((seat, m.start(), m.end()))
        for pattern, pos in self._name_patterns:
            for m in pattern.finditer(text):
                if not any(s <= m.start() < e for _, s, e in found):
                    found.append((pos, m.start(), m.end()))
        return sorted(found, key=lambda t: t[1])


def _walk(obj: Any, skip: frozenset[str], depth: int = 0):
    """Yield (key, value) for every dict entry under ``obj`` (not under ``skip``)."""
    if depth > 5:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip:
                continue
            yield k, v
            yield from _walk(v, skip, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, skip, depth + 1)


_PUBLIC_KEYS = frozenset({"game_state", "players", "new_messages", "legal_actions", "next_actions", "returns"})


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


_ROLE_IN_OBS = re.compile(
    r"(?:your\s+role\s*(?:is|:)?|you\s+are\s+(?:a|an|the)|role\s*[:=])\s*(?:a|an|the)?\s*(werewolf|wolf|seer|villager)",
    re.I,
)
_RESULT_IN_OBS = re.compile(
    r"player\s*[_#-]?\s*(\d+)\b[^.\n;]*?\b(not\s+a\s+(?:were)?wolf|werewolf|wolf|villager|innocent|good|evil|seer)\b",
    re.I,
)
_PARTNER_HINT = re.compile(r"partner|teammate|ally|allies|fellow|other\s+(?:were)?wolf|wolves?\s+(?:are|is)|pack", re.I)
_EVENT_HINT = re.compile(r"kill|died|dead|eliminat|lynch|voted|investigat|executed", re.I)
_INVESTIGATIONS_LINE = re.compile(r"\s*your\s+investigations?\s*:", re.I)
_SEER_HINT = re.compile(r"invest|check|inspect|reveal|result|saw|seer|vision|peek", re.I)


def parse_private(state_raw: dict, observation: str, me: int, resolver: Resolver) -> tuple[str | None, set[int], dict[int, str]]:
    """(my role, my wolf partners, my seer results) from the private view."""
    role: str | None = None
    partners: set[int] = set()
    results: dict[int, str] = {}

    for key, value in _walk(state_raw, _PUBLIC_KEYS):
        k = str(key).lower()
        if k in ("role", "your_role", "my_role") and role is None:
            role = normalize_role(value) if isinstance(value, str) else None
        elif k in ("wolves", "teammates", "partners", "allies", "fellow_wolves", "known_wolves", "wolf_team"):
            items = value if isinstance(value, list) else [value]
            for item in items:
                seat = resolver.seat(item)
                if seat is not None and seat != me:
                    partners.add(seat)
        elif k in ("investigations", "seer_results", "investigation_results", "inspections", "checks"):
            if isinstance(value, dict):
                for target, verdict in value.items():
                    seat, verdict_role = resolver.seat(target), normalize_role(str(verdict))
                    if seat is not None and verdict_role:
                        results[seat] = WOLF if verdict_role == WOLF else VILLAGER
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        seat = resolver.seat(item.get("target", item.get("player")))
                        verdict_role = normalize_role(str(item.get("result", item.get("role", ""))))
                        if seat is not None and verdict_role:
                            results[seat] = WOLF if verdict_role == WOLF else VILLAGER

    if role is None:
        m = _ROLE_IN_OBS.search(observation)
        if m:
            role = normalize_role(m.group(1))

    if role == WOLF and not partners:
        for sentence in _sentences(observation):
            if _PARTNER_HINT.search(sentence) and not _EVENT_HINT.search(sentence):
                for seat, _, _ in resolver.mentions(sentence):
                    if seat != me:
                        partners.add(seat)

    if role == SEER:
        # The platform lists results on one line ("Your investigations: Player0 is
        # wolf, ..."); reading only that line keeps the public death list
        # ("Player2 was villager (killed at night)") from passing as a check.
        own_line = [line for line in observation.splitlines() if _INVESTIGATIONS_LINE.match(line)]
        for sentence in own_line or _sentences(observation):
            if not _SEER_HINT.search(sentence) and not re.search(r"\bis\b|[:=]|->", sentence):
                continue
            for m in _RESULT_IN_OBS.finditer(sentence):
                seat = int(m.group(1))
                verdict = normalize_role(m.group(2))
                if seat != me and 0 <= seat < resolver.n and verdict and seat not in results:
                    results[seat] = WOLF if verdict == WOLF else VILLAGER

    return role, partners, results


def parse_dead(game_state: dict, resolver: Resolver) -> dict[int, Death]:
    dead: dict[int, Death] = {}
    entries = game_state.get("dead") or []
    if isinstance(entries, dict):
        entries = [{"name": k, **(v if isinstance(v, dict) else {"role": v})} for k, v in entries.items()]
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict):
            seat = None
            for key in ("name", "player", "position", "seat", "player_name"):
                if key in entry:
                    seat = resolver.seat(entry[key])
                    if seat is not None:
                        break
            if seat is None:
                continue
            role = normalize_role(str(entry.get("role", entry.get("true_role", entry.get("team", "")))))
            cause = entry.get("cause", entry.get("reason", entry.get("tag")))
            day = entry.get("day", entry.get("round"))
            dead[seat] = Death(seat, role, str(cause) if cause is not None else None, day if isinstance(day, int) else None)
        else:
            seat = resolver.seat(entry)
            if seat is not None:
                dead[seat] = Death(seat, None, None, None)
    return dead


_VOTER_KEYS = ("voter", "from", "player", "source", "by", "name")
_TARGET_KEYS = ("target", "vote", "voted_for", "to", "choice", "against")


def parse_votes(history: Any, resolver: Resolver) -> list[Vote]:
    votes: list[Vote] = []

    def target_seat(value: Any) -> int | None:
        if value in (ABSTAIN, str(ABSTAIN)) or (isinstance(value, str) and "abstain" in value.lower()):
            return None
        return resolver.seat(value)

    def handle(entry: Any, day: int | None) -> None:
        if isinstance(entry, list):
            for item in entry:
                handle(item, day)
            return
        if not isinstance(entry, dict):
            return
        d = entry.get("day", entry.get("round", day))
        d = d if isinstance(d, int) else day
        inner = entry.get("votes", entry.get("ballots"))
        if isinstance(inner, dict):
            for voter, target in inner.items():
                v = resolver.seat(voter)
                if v is not None:
                    votes.append(Vote(d, v, target_seat(target)))
            return
        if isinstance(inner, list):
            handle(inner, d)
            return
        voter_key = next((k for k in _VOTER_KEYS if k in entry), None)
        target_key = next((k for k in _TARGET_KEYS if k in entry), None)
        if voter_key and target_key:
            v = resolver.seat(entry[voter_key])
            if v is not None:
                votes.append(Vote(d, v, target_seat(entry[target_key])))
            return
        # {"1": {...}} keyed by day, or {"Player3": "Player5"} keyed by voter.
        for k, val in entry.items():
            if isinstance(k, str) and k.isdigit() and isinstance(val, (dict, list)):
                handle(val, int(k))
            else:
                v = resolver.seat(k)
                if v is not None and not isinstance(val, (dict, list)):
                    votes.append(Vote(d, v, target_seat(val)))

    handle(history, None)
    return votes


def snapshot(state: Any, seat_hint: int | None = None) -> tuple[Snapshot, Resolver]:
    raw: dict = state.raw if isinstance(getattr(state, "raw", None), dict) else {}
    players = [p for p in raw.get("players") or [] if isinstance(p, dict)]
    n = max([7] + [p["position"] + 1 for p in players if isinstance(p.get("position"), int)])
    resolver = Resolver(players, n)
    me = raw.get("your_position")
    if not isinstance(me, int):
        me = seat_hint if isinstance(seat_hint, int) else 0
    gs = raw.get("game_state") if isinstance(raw.get("game_state"), dict) else {}

    dead = parse_dead(gs, resolver)
    alive_raw = gs.get("alive")
    if isinstance(alive_raw, list) and alive_raw:
        alive = {s for s in (resolver.seat(a) for a in alive_raw) if s is not None}
    else:
        alive = set(range(n)) - set(dead)
    alive -= set(dead)

    role, partners, results = parse_private(raw, str(getattr(state, "observation", "") or ""), me, resolver)
    day = gs.get("day") if isinstance(gs.get("day"), int) else 0
    seconds = raw.get("messaging_seconds_left")
    return (
        Snapshot(
            me=me,
            n_players=n,
            names={p["position"]: str(p.get("name")) for p in players if isinstance(p.get("position"), int)},
            alive=alive,
            dead=dead,
            role=role,
            partners=partners,
            seer_results=results,
            day=day,
            sub_phase=gs.get("phase"),
            votes=parse_votes(gs.get("vote_history") or gs.get("votes") or [], resolver),
            seconds_left=float(seconds) if isinstance(seconds, (int, float)) else None,
        ),
        resolver,
    )


# -- chat -------------------------------------------------------------------------------------

# First-person seer claims only. "Player0's seer claim makes sense" is not a claim
# (live games showed that wording turning every commenter into a "seer").
_SEER_CLAIM = re.compile(
    r"\b(?:i\s*am|i'm|im|i\s+was)\s+(?:the\s+|a\s+|your\s+|real\s+|actual\s+|true\s+)*seer\b"
    r"|(?:^|[.!?]\s*)(?:seer\s+here|as\s+(?:the\s+)?seer)\b|\bi\s+claim\s+seer\b",
    re.I,
)
_SEER_DENY = re.compile(r"\b(?:i\s*am|i'm|im)\s+not\s+(?:the\s+|a\s+)?seer\b", re.I)
_VILLAGER_CLAIM = re.compile(
    r"\b(?:i\s*am|i'm|im)\s+(?:a\s+|just\s+a\s+|only\s+a\s+|the\s+)*(?:plain\s+|simple\s+)?villager\b"
    r"|\b(?:just|only|simply)\s+a\s+(?:plain\s+|simple\s+)?villager\b|\bvillager\s+here\b",
    re.I,
)
_WOLF_WORDS = re.compile(r"(?:were)?wol(?:f|ves)\b|\bevil\b|\bliar\b|\blying\b|\bsuspicious\b|\bsus\b|\bfake\b", re.I)
_CLEAR_WORDS = re.compile(r"\bvillager\b|\binnocent\b|\bclean\b|\bcleared?\b|\bgood\b|\btrust(?:ed|worthy)?\b|\bnot\s+(?:a\s+)?(?:were)?wolf\b|\bsafe\b|\btown\b", re.I)
_VOTE_BEFORE = re.compile(
    r"(?:(?:vote|voting|lynch|eliminate|execute|suspect|accuse|kill)\s*(?:for|out|on|against)?|tells\s+me\s+it[’']?s)\s*$",
    re.I,
)
_RESULT_BEFORE = re.compile(r"(?:check(?:ed)?|investigat\w*|inspect\w*|look(?:ed)?\s+at|saw|peek\w*)\s*(?:on)?\s*$", re.I)
_NEGATION = re.compile(r"\b(?:not|n't|never|no)\b", re.I)
# "avoid a random vote on Player0", "why mention avoiding a vote on Player0", "rather than voting
# Player0": talking about a vote, not casting one (a live game misread both as votes on us and
# our agent fired back denials at two players who had defended it).
_VOTE_DISCUSSED = re.compile(
    r"\b(?:avoid\w*|against|instead\s+of|rather\s+than|without|hold(?:ing)?\s+off|refrain\w*|oppos\w*|"
    r"mention\w*|propos\w*|suggest\w*\s+not|careful|random)\b", re.I)
# A question about someone's vote ("why vote Player0?", "who proposed voting Player0?") isn't one.
_ASKING = re.compile(r"^\W*(?:why|what|how|who|when|did|does|do|is|are|was|were)\b[^.!]*\?", re.I)
# "Player0 has identified them as a wolf", "Player6 claims...": the name is the subject
# reporting something, not the one being called a wolf.
_REPORTING_AFTER = re.compile(
    r"^\s*(?:['’]s\s+)?(?:has|have|had|just)?\s*(?:identified|found|checked|named|called|said|says|claims?|claimed"
    r"|accused|accuses|thinks?|believes?|suspects?|counter-?claimed|pointed|noted|asked)\b",
    re.I,
)
# "I support Player0", "trust Player0", "backing Player0": an endorsement, never an accusation.
_SUPPORT_BEFORE = re.compile(
    r"(?:support(?:ing)?|trust(?:ing)?|back(?:ing)?|believ(?:e|ing)|agree(?:ing)?\s+with|side\s+with|defend(?:ing)?)\s+$",
    re.I,
)
# Text in quotes is someone else's words ("both say 'confident in my vote for Player4'").
_QUOTED = re.compile(r"[\"“][^\"”]{3,}?[\"”]|(?<=[\s:(])['‘][^'’]{3,}?['’](?=[\s.,;:!?)]|$)")
# A claimed result reads "Player3 is a wolf", "Player3 = WOLF", "Player3: villager".
_RESULT_AFTER = re.compile(
    r"^\s*(?:is|was|=|:|->|=>)\s*(?:definitely\s+|a\s+|an\s+)*(?:not\s+(?:a\s+)?)?"
    r"(?:were)?(?:wol(?:f|ves)|villager|innocent|good|clean|evil|town)\b",
    re.I,
)


@dataclass
class ChatFacts:
    """What the table has said, accumulated across days."""

    seer_claims: dict[int, int] = field(default_factory=dict)  # claimant -> day of first claim
    claimed_results: dict[int, dict[int, str]] = field(default_factory=dict)  # claimant -> {target: role}
    # claimant -> earlier claimants who had named them a wolf before they claimed
    named_before_claim: dict[int, set[int]] = field(default_factory=dict)
    villager_claims: set[int] = field(default_factory=set)
    accusations: list[tuple[int, int, int]] = field(default_factory=list)  # (speaker, target, day)
    clears: list[tuple[int, int, int]] = field(default_factory=list)  # (speaker, target, day)
    vote_intents: dict[int, dict[int, int]] = field(default_factory=dict)  # day -> {speaker: target}
    message_counts: dict[int, int] = field(default_factory=dict)
    private_from: list[tuple[int, str, int]] = field(default_factory=list)  # (sender, text, day) sent only to me
    # Unprompted vote announcements, in order: (speaker, target, day, public message number).
    # "Unprompted": not joining someone ("joining PlayerY's vote"), not a "psychic" pick,
    # and not a target a seer claimant called a wolf.
    pushes: list[tuple[int, int, int, int]] = field(default_factory=list)
    # Two players who pushed the same target within UNISON_GAP public messages of each other,
    # each for the first time: (first, second, target, day).
    unison: list[tuple[int, int, int, int]] = field(default_factory=list)
    public_messages: int = 0
    # Someone asked me something by name ("Player3, what do you think?"): (sender, day).
    asked_me: list[tuple[int, int]] = field(default_factory=list)

    def intents_on(self, day: int) -> dict[int, int]:
        return self.vote_intents.get(day, {})


UNISON_GAP = 2
_ASK_SPLIT = re.compile(r"(?<=[.!?\n;])")
_ASK_WORDS = re.compile(
    r"\?|\b(?:thoughts|opinion|your\s+(?:read|take|view)|what\s+do\s+you\s+think|explain|why\s+(?:did|are|would)\s+you"
    r"|care\s+to|defend|weigh\s+in|who\s+do\s+you)\b",
    re.I,
)
_FOLLOW = re.compile(
    r"\bjoin(?:ing|s)?\b|\bagree\b|\bsecond(?:ing)?\b|\bwith\s+player\s*\d|\bpsychic\b|\bback(?:ing|s)?\b"
    r"|\bsupport(?:ing|s)?\b|\bfollow(?:ing|s)?\b|\bbehind\b|\bon\s+board\b"
    r"|player\s*\d+\s*['’]s\s+(?:vote|call|push|pick|lead|choice)",
    re.I,
)


def _record_push(facts: ChatFacts, sender: int, target: int, day: int, text: str) -> None:
    if _FOLLOW.search(text):
        return  # following someone else, or our own shared day-1 pick
    if any(results.get(target) == WOLF for results in facts.claimed_results.values()):
        return  # piling onto a seer claimant's "wolf" is natural, not coordination
    if any(s == sender and t == target and d == day for s, t, d, _ in facts.pushes):
        return  # only the first push counts
    for other, t, d, n in reversed(facts.pushes):
        if d != day or facts.public_messages - n > UNISON_GAP:
            break
        if t == target and other != sender and other not in facts.seer_claims and sender not in facts.seer_claims:
            facts.unison.append((other, sender, target, day))
            break
    facts.pushes.append((sender, target, day, facts.public_messages))


_PLAIN_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201b": "'", "\u00b4": "'", "`": "'",
                                "\u201c": '"', "\u201d": '"', "\u201e": '"'})


_TEAM_SUFFIX = re.compile(r"\s*\((?:Player\s*\d+\s*,\s*)?[^()]*\)\s*$")


def team_name(display_name: Any, limit: int = 40) -> str:
    """The team behind a seat's display name ("Yuriko Red (Player2, Self-hosted)" -> "Yuriko Red"),
    as plain printable text of bounded length: other contestants choose these names, so they're
    data, never instructions."""
    name = _TEAM_SUFFIX.sub("", str(display_name or ""))
    name = "".join(ch for ch in name if ch.isprintable() and ch not in "\"`<>{}[]").strip()
    return name[:limit].strip() or "?"


def seat_teams(raw: Any) -> dict[int, str]:
    """{seat: team name} from a state's ``players`` list (empty when the server sends none)."""
    players = raw.get("players") if isinstance(raw, dict) else None
    out = {}
    for p in players or []:
        if isinstance(p, dict) and isinstance(p.get("position"), int) and p.get("name"):
            out[p["position"]] = team_name(p["name"])
    return out


def plain_quotes(text: str) -> str:
    """Curly quotes as straight ones, so our patterns read "I’m the Seer" (as OpenAI models write it)
    the same as "I'm the Seer"."""
    return text.translate(_PLAIN_QUOTES)


def read_message(facts: ChatFacts, sender: int, text: str, recipients: list[int], day: int, me: int, resolver: Resolver) -> None:
    text = plain_quotes(text)
    facts.message_counts[sender] = facts.message_counts.get(sender, 0) + 1
    if not recipients:
        facts.public_messages += 1
    if recipients and sender != me:
        facts.private_from.append((sender, text, day))
    if sender == me:
        return

    if _SEER_CLAIM.search(text) and not _SEER_DENY.search(text) and sender not in facts.seer_claims:
        facts.seer_claims[sender] = day
        # Who had already "found" this player to be a wolf before they claimed?
        facts.named_before_claim[sender] = {
            c for c, results in facts.claimed_results.items() if results.get(sender) == WOLF and c != sender
        }
    if _VILLAGER_CLAIM.search(text):
        facts.villager_claims.add(sender)

    is_claimant = sender in facts.seer_claims
    unquoted = _QUOTED.sub(" ", text)
    if any(_ASK_WORDS.search(sentence) and any(seat == me for seat, _, _ in resolver.mentions(sentence))
           for sentence in _ASK_SPLIT.split(unquoted)):
        facts.asked_me.append((sender, day))
    for sentence in _sentences(unquoted):
        mentions = resolver.mentions(sentence)
        for i, (seat, start, end) in enumerate(mentions):
            if seat == sender:
                continue
            nxt = mentions[i + 1][1] if i + 1 < len(mentions) else len(sentence)
            after = sentence[end:nxt][:60]
            before = sentence[max(0, start - 40):start]
            wolfish = bool(_WOLF_WORDS.search(after))
            clearish = bool(_CLEAR_WORDS.search(after))
            # "Player3, what do you think?" / "Player3, Player5: one of you..." addresses them. Only
            # at the start of a sentence or in a list of names there, never "I checked Player2: wolf".
            lead_in = sentence[:start] if i == 0 else sentence[mentions[i - 1][2]:start]
            in_address_position = lead_in.strip(" ,:") in ("", "and", "or") and (i == 0 or not before.strip(" ,:").endswith(":"))
            vocative = (in_address_position and re.match(r"^\s*[,:]", after) is not None
                        and not re.search(r"\byou", after[:20], re.I))
            if vocative or _REPORTING_AFTER.search(after):
                wolfish = clearish = False  # addressing them, or reporting what they said
            if _SUPPORT_BEFORE.search(before):
                wolfish, clearish = False, True
            negated = bool(_NEGATION.search(after.split(",")[0])) if (wolfish or clearish) else False
            if wolfish and negated and not clearish:
                wolfish, clearish = False, True
            vote_target = bool(_VOTE_BEFORE.search(before))
            if is_claimant and (wolfish or clearish) and (_RESULT_BEFORE.search(before) or _RESULT_AFTER.search(after)):
                verdict = WOLF if wolfish and not clearish else VILLAGER
                facts.claimed_results.setdefault(sender, {})[seat] = verdict
            if vote_target and not _NEGATION.search(before[-15:]) and not _VOTE_DISCUSSED.search(before[-35:]) \
                    and not _ASKING.match(sentence):
                facts.vote_intents.setdefault(day, {})[sender] = seat
                facts.accusations.append((sender, seat, day))
                if not recipients:
                    _record_push(facts, sender, seat, day, text)
            elif wolfish and not clearish:
                facts.accusations.append((sender, seat, day))
            elif clearish and not wolfish:
                facts.clears.append((sender, seat, day))
