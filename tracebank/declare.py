"""DECLARE constraints as a divergence signal (experiment E22).

Everything else in this repo compares a running prefix to reference traces as a
STRING (edit distance, set Jaccard) or as a WALK (return rate, edge novelty).
A DECLARE model is neither: it is a set of temporal-logic constraints, each of
which the prefix either satisfies, violates, or has not yet decided.

THE PROBLEM THIS FILE SOLVES.  On a finished trace a constraint is simply true
or false.  On a PREFIX it need not be either: `response(a,b)` -- "every a is
eventually followed by a b" -- is never permanently violated by any prefix,
because the b may still arrive.  It is *pending*.  Scoring a prefix as though
it were finished would report a violation every time an obligation is merely
outstanding, which on short prefixes is almost always.

So each constraint is monitored with the four-valued RV-LTL state:

    PERM_VIOL  no extension of this prefix satisfies the constraint  (absorbing)
    POSS_VIOL  the prefix does not satisfy it, but an extension could -- an
               OUTSTANDING OBLIGATION.  This is the state a string metric
               cannot express.
    POSS_SAT   the prefix satisfies it, but an extension could break it
    PERM_SAT   every extension satisfies it                          (absorbing)

Of the 19 templates, 14 can reach PERM_VIOL online and 5 can only ever go
pending (`end`, `existence`, `response`, `coexistence`, `choice`).  That split
is why permanent violations alone are too sparse to score a 12-step prefix, and
why the pending count is carried as a separate feature rather than folded in.

WITNESSES ARE TRACKED SEPARATELY, and deliberately so.  A constraint mined from
FAILING runs says "failing runs do this"; confirming it online would need
PERM_SAT, which only 4 of the 19 templates can ever reach, so the negative model
would be silent.  It is scored instead by existential witnesses -- has the
prefix produced ONE instance of the pattern?  That is a property of the prefix
alone and must keep updating after the constraint itself has been permanently
violated, so witness tracking cannot live inside the (absorbing) monitor.
`b a b` witnesses the bigram of `chain_precedence(a,b)` even though the leading
`b` already killed the constraint.

SEMANTICS ARE THOSE OF THE SIESTA MINER, deliberately.  The constraints are
discovered by SIESTA (/home/neron/lab/siesta-framework), so the checker must
agree with it or the mined support figures describe a different property from
the one being monitored.  Two consequences, both verified against the source:

  * `coexistence(a,b)` is "both occur", NOT the usual biconditional
    (siesta/modules/mine/unordered.py:43).
  * ordered templates use TIGHT semantics -- the relation must hold for ALL
    occurrences of the pair -- and are only mined on traces where both
    activities occur (ordered.py:41-45, non-vacuous support).

The batch functions `satisfies()` and `witnesses()` at the bottom are
INDEPENDENT implementations, written from the index-based definitions rather
than from the monitors, and are what tests/test_declare.py checks the streaming
monitors against at every prefix of every random trace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #

PERM_VIOL, POSS_VIOL, POSS_SAT, PERM_SAT = 0, 1, 2, 3
STATE_NAMES = {PERM_VIOL: "perm_viol", POSS_VIOL: "poss_viol",
               POSS_SAT: "poss_sat", PERM_SAT: "perm_sat"}
ABSORBING = (PERM_VIOL, PERM_SAT)

#: states in which a *finished* trace fails the constraint
FAILING = (PERM_VIOL, POSS_VIOL)

UNARY = ("init", "end", "existence", "absence", "exactly")
BINARY = ("response", "precedence", "succession", "alternate_response",
          "alternate_precedence", "chain_response", "chain_precedence",
          "chain_succession", "not_succession", "not_chain_succession",
          "coexistence", "choice", "exclusive_choice", "not_coexistence")
TEMPLATES = UNARY + BINARY

#: SIESTA's category for each template (used to reproduce its support formulas)
CATEGORY = {
    "init": "positional", "end": "positional",
    "existence": "existential", "absence": "existential", "exactly": "existential",
    "response": "ordered", "precedence": "ordered", "succession": "ordered",
    "alternate_response": "ordered", "alternate_precedence": "ordered",
    "chain_response": "ordered", "chain_precedence": "ordered",
    "chain_succession": "ordered", "not_succession": "ordered",
    "not_chain_succession": "ordered",
    "coexistence": "unordered", "choice": "unordered",
    "exclusive_choice": "unordered",
    "not_coexistence": "negation",
}

#: templates whose satisfaction a prefix can CONFIRM by a single instance.
#: A universal negative ("a never precedes b") cannot be witnessed by any
#: prefix, so the negative model of E22 is restricted to these.
WITNESS_TEMPLATES = ("init", "existence", "response", "precedence",
                     "succession", "alternate_response",
                     "alternate_precedence", "chain_response",
                     "chain_precedence", "chain_succession", "coexistence")

#: templates that are conjunctions of two others already in the model.  Counting
#: them as well double-counts a violation on exactly the pairs where both
#: survive selection, and that inflation is not constant across runs.
DERIVED = ("succession", "chain_succession")

#: `exclusive_choice` and `not_coexistence` have identical permanent-violation
#: indicators (both activities seen), so only one may enter a model.
#: `end` and `choice` are near-tautological over an 18-symbol tool alphabet.
DEFAULT_EXCLUDED = DERIVED + ("not_coexistence", "end", "choice")

#: Templates that count occurrences of a single activity.  On tau-bench these
#: dominate discriminative selection, and what they select is run length wearing
#: a DECLARE hat: `absence(user,15)` is "fewer than 15 user turns" and
#: `existence(user,3)` is "at least 3".  Length is the confound this whole
#: project exists to control for, so E22 reports a model with and a model
#: without them -- see `dcl_ord_*` in experiments/e22_declare.py.
COUNTING_TEMPLATES = ("existence", "absence", "exactly")

#: Which labels ACTIVATE a constraint -- i.e. make its antecedent fire, so
#: that it now has something to say about this trace.  Standard DECLARE
#: activation: `response(a,b)` is activated by an `a`, `precedence(a,b)` by a
#: `b`.  NOTE the consequence, which E24 measures directly: activation depends
#: only on WHICH labels occurred, never on their order.
ACTIVATED_BY = {
    "init": "any", "end": "any",
    "existence": "src", "absence": "src", "exactly": "src",
    "response": "src", "alternate_response": "src", "chain_response": "src",
    "not_succession": "src", "not_chain_succession": "src",
    "precedence": "tgt", "alternate_precedence": "tgt",
    "chain_precedence": "tgt",
    "succession": "either", "chain_succession": "either",
    "coexistence": "either", "choice": "either",
    "exclusive_choice": "either", "not_coexistence": "either",
}


def activation_labels(c: "Constraint"):
    """The labels whose occurrence activates this constraint.

    Returns () for templates activated by any event at all (`init`, `end`).
    """
    kind = ACTIVATED_BY[c.template]
    if kind == "any":
        return ()
    if kind == "src":
        return (c.source,)
    if kind == "tgt":
        return (c.target,)
    return tuple(x for x in (c.source, c.target) if x is not None)


#: Templates that constrain ORDER rather than multiplicity.  A violation here is
#: a statement about the shape of the run, which is what the paper claims to
#: measure.
ORDERING_TEMPLATES = tuple(
    t for t in TEMPLATES
    if t not in COUNTING_TEMPLATES and t not in ("init", "end"))


@dataclass(frozen=True)
class Constraint:
    """One mined constraint: the template, its parameters, and its statistics."""
    template: str
    source: str
    target: Optional[str] = None
    n: Optional[int] = None
    support: float = 0.0
    confidence: float = 0.0
    interest: Optional[float] = None
    weight: float = 1.0          # set by select_discriminative (|delta support|)

    @property
    def category(self) -> str:
        return CATEGORY[self.template]

    @property
    def key(self) -> Tuple:
        return (self.template, self.source, self.target, self.n)

    def __str__(self) -> str:
        if self.target is None:
            arg = f"{self.source}" + (f",{self.n}" if self.n is not None else "")
        else:
            arg = f"{self.source},{self.target}"
        return f"{self.template}({arg})"


# --------------------------------------------------------------------------- #
# Monitors -- one per template, O(1) state and O(1) update
# --------------------------------------------------------------------------- #

class Monitor:
    """Streaming RV-LTL monitor for one constraint.

    `push(lab, prev)` is called only when the event is RELEVANT to this monitor,
    which DeclareState decides from the WAKE code below.
    A monitor that is not woken cannot change state, so the aggregate counters
    DeclareState maintains stay correct without rescanning.
    """
    __slots__ = ("c", "a", "b", "n", "state", "_act")

    #: Which events can possibly change this monitor's state.  A class-level
    #: code rather than instance attributes, because __slots__ exists to keep
    #: these objects small -- there is one per constraint per live session.
    #:
    #:   always      every event            (`init` needs step 1; `end` flips)
    #:   src / tgt   the event equals the source / target
    #:   pair        the event equals either
    #:   *+after     ALSO the event immediately following a source, whatever it
    #:               is -- the chain templates must see what came next
    WAKE = "none"

    def __init__(self, c: Constraint):
        self.c = c
        self.a = c.source
        self.b = c.target
        self.n = c.n
        self.state = POSS_VIOL
        self._act = False

    def _update(self, x: str, prev: Optional[str]) -> None:
        raise NotImplementedError

    def push(self, x: str, prev: Optional[str]) -> int:
        if self.state in ABSORBING:
            return self.state
        self._update(x, prev)
        return self.state

    @property
    def activated(self) -> bool:
        """Has the antecedent fired?  Set by DeclareState, not here: several
        templates are not woken by their own activating label (e.g.
        not_chain_succession(a,b) wakes on `b` but is activated by `a`)."""
        return self._act


def wake_sets(c: Constraint) -> Tuple[Tuple, Tuple, bool]:
    """(labels, after_labels, always) for the constraint's monitor."""
    code = MONITORS[c.template].WAKE
    always = code == "always"
    after = (c.source,) if code.endswith("+after") else ()
    head = code.split("+")[0]
    if head == "src":
        labels = (c.source,)
    elif head == "tgt":
        labels = (c.target,)
    elif head == "pair":
        labels = (c.source, c.target)
    else:
        labels = ()
    return labels, after, always


# ---- unary ---------------------------------------------------------------- #

class _Init(Monitor):
    """init(a): the first event is `a`.  Decided permanently at step 1."""
    __slots__ = ("_seen",)
    WAKE = "always"

    def __init__(self, c):
        super().__init__(c)
        self._seen = False

    def _update(self, x, prev):
        if self._seen:
            return
        self._seen = True
        self.state = PERM_SAT if x == self.a else PERM_VIOL


class _End(Monitor):
    """end(a): the last event is `a`.  Never decidable before the trace ends."""
    __slots__ = ()
    WAKE = "always"

    def _update(self, x, prev):
        self.state = POSS_SAT if x == self.a else POSS_VIOL


class _Existence(Monitor):
    """existence(a,n): `a` occurs at least n times.  Only ever pending or
    permanently satisfied -- a count never decreases."""
    __slots__ = ("_cnt",)
    WAKE = "src"

    def __init__(self, c):
        super().__init__(c)
        self._cnt = 0

    def _update(self, x, prev):
        if x != self.a:
            return
        self._cnt += 1
        if self._cnt >= (self.n or 1):
            self.state = PERM_SAT


class _Absence(Monitor):
    """absence(a,n): `a` occurs fewer than n times.  Cannot be repaired."""
    __slots__ = ("_cnt",)
    WAKE = "src"

    def __init__(self, c):
        super().__init__(c)
        self._cnt = 0
        self.state = POSS_SAT

    def _update(self, x, prev):
        if x != self.a:
            return
        self._cnt += 1
        self.state = PERM_VIOL if self._cnt >= (self.n or 1) else POSS_SAT


class _Exactly(Monitor):
    """exactly(a,n): `a` occurs exactly n times.  Pending below n, permanently
    violated above it."""
    __slots__ = ("_cnt",)
    WAKE = "src"

    def __init__(self, c):
        super().__init__(c)
        self._cnt = 0
        self.state = POSS_SAT if (c.n or 0) == 0 else POSS_VIOL

    def _update(self, x, prev):
        if x != self.a:
            return
        self._cnt += 1
        n = self.n or 0
        if self._cnt > n:
            self.state = PERM_VIOL
        elif self._cnt == n:
            self.state = POSS_SAT
        else:
            self.state = POSS_VIOL


# ---- ordered -------------------------------------------------------------- #

class _Response(Monitor):
    """response(a,b): every `a` is eventually followed by a `b`.

    The canonical pending template: an unmatched `a` leaves an obligation that
    no prefix can refute.  Discharge before opening, so that with a == b the
    obligation correctly re-opens on every occurrence.
    """
    __slots__ = ("_open",)
    WAKE = "pair"

    def __init__(self, c):
        super().__init__(c)
        self._open = False
        self.state = POSS_SAT

    def _update(self, x, prev):
        if x == self.b:
            self._open = False
        if x == self.a:
            self._open = True
        self.state = POSS_VIOL if self._open else POSS_SAT


class _Precedence(Monitor):
    """precedence(a,b): every `b` is preceded by an earlier `a`.

    Permanently violated by a `b` with no prior `a`; permanently SATISFIED once
    an `a` has been seen without a prior offence, because every later `b` is
    then covered.
    """
    __slots__ = ("_seen_a",)
    WAKE = "pair"

    def __init__(self, c):
        super().__init__(c)
        self._seen_a = False
        self.state = POSS_SAT

    def _update(self, x, prev):
        if x == self.b and not self._seen_a:
            self.state = PERM_VIOL
            return
        if x == self.a:
            self._seen_a = True
            self.state = PERM_SAT


class _Succession(Monitor):
    """succession(a,b) = response(a,b) AND precedence(a,b)."""
    __slots__ = ("_open", "_seen_a")
    WAKE = "pair"

    def __init__(self, c):
        super().__init__(c)
        self._open = False
        self._seen_a = False
        self.state = POSS_SAT

    def _update(self, x, prev):
        if x == self.b:
            if not self._seen_a:
                self.state = PERM_VIOL
                return
            self._open = False
        if x == self.a:
            self._seen_a = True
            self._open = True
        self.state = POSS_VIOL if self._open else POSS_SAT


class _AlternateResponse(Monitor):
    """alternate_response(a,b): every `a` is followed by a `b` before the next
    `a`.  A second `a` while an obligation is open can never be repaired."""
    __slots__ = ("_open",)
    WAKE = "pair"

    def __init__(self, c):
        super().__init__(c)
        self._open = False
        self.state = POSS_SAT

    def _update(self, x, prev):
        if self._open and x == self.a:
            self.state = PERM_VIOL
            return
        if x == self.b:
            self._open = False
        if x == self.a:
            self._open = True
        self.state = POSS_VIOL if self._open else POSS_SAT


class _AlternatePrecedence(Monitor):
    """alternate_precedence(a,b): every `b` is preceded by an `a` that comes
    after the previous `b`.  A `b` without credit is unrepairable."""
    __slots__ = ("_credit",)
    WAKE = "pair"

    def __init__(self, c):
        super().__init__(c)
        self._credit = False
        self.state = POSS_SAT

    def _update(self, x, prev):
        if x == self.b:
            if not self._credit:
                self.state = PERM_VIOL
                return
            self._credit = False
        if x == self.a:
            self._credit = True
        self.state = POSS_SAT


class _ChainResponse(Monitor):
    """chain_response(a,b): every `a` is IMMEDIATELY followed by `b`.

    Needs waking on the event after an `a` whatever that event is, which is what
    the "+after" WAKE code exists for.  Pending when the prefix ends on an `a`.
    """
    __slots__ = ("_last_is_a",)
    WAKE = "src+after"

    def __init__(self, c):
        super().__init__(c)
        self._last_is_a = False
        self.state = POSS_SAT

    def _update(self, x, prev):
        if prev == self.a and x != self.b:
            self.state = PERM_VIOL
            return
        self._last_is_a = (x == self.a)
        self.state = POSS_VIOL if self._last_is_a else POSS_SAT


class _ChainPrecedence(Monitor):
    """chain_precedence(a,b): every `b` is IMMEDIATELY preceded by `a`.
    A `b` at position 1 has no predecessor and violates."""
    __slots__ = ()
    WAKE = "tgt"

    def __init__(self, c):
        super().__init__(c)
        self.state = POSS_SAT

    def _update(self, x, prev):
        if x == self.b and prev != self.a:
            self.state = PERM_VIOL


class _ChainSuccession(Monitor):
    """chain_succession(a,b) = chain_response AND chain_precedence."""
    __slots__ = ("_last_is_a",)
    WAKE = "pair+after"

    def __init__(self, c):
        super().__init__(c)
        self._last_is_a = False
        self.state = POSS_SAT

    def _update(self, x, prev):
        if prev == self.a and x != self.b:
            self.state = PERM_VIOL
            return
        if x == self.b and prev != self.a:
            self.state = PERM_VIOL
            return
        self._last_is_a = (x == self.a)
        self.state = POSS_VIOL if self._last_is_a else POSS_SAT


class _NotSuccession(Monitor):
    """not_succession(a,b): no `a` ever occurs before a `b`.  A universal
    negative -- refutable by one witness, never confirmable by a prefix."""
    __slots__ = ("_seen_a",)
    WAKE = "pair"

    def __init__(self, c):
        super().__init__(c)
        self._seen_a = False
        self.state = POSS_SAT

    def _update(self, x, prev):
        if x == self.b and self._seen_a:
            self.state = PERM_VIOL
            return
        if x == self.a:
            self._seen_a = True


class _NotChainSuccession(Monitor):
    """not_chain_succession(a,b): `a` is never immediately followed by `b`."""
    __slots__ = ()
    WAKE = "tgt"

    def __init__(self, c):
        super().__init__(c)
        self.state = POSS_SAT

    def _update(self, x, prev):
        if x == self.b and prev == self.a:
            self.state = PERM_VIOL


# ---- unordered / negation ------------------------------------------------- #

class _Pair(Monitor):
    """Shared state for the four set-membership templates."""
    __slots__ = ("_sa", "_sb")
    WAKE = "pair"

    def __init__(self, c):
        super().__init__(c)
        self._sa = self._sb = False

    def _mark(self, x):
        if x == self.a:
            self._sa = True
        if x == self.b:
            self._sb = True


class _CoExistence(_Pair):
    """coexistence(a,b): BOTH occur.  SIESTA's reading, not the biconditional
    (unordered.py:43).  Pending until both are seen, then permanent."""
    __slots__ = ()

    def _update(self, x, prev):
        self._mark(x)
        if self._sa and self._sb:
            self.state = PERM_SAT


class _Choice(_Pair):
    """choice(a,b): at least one occurs."""
    __slots__ = ()

    def _update(self, x, prev):
        self._mark(x)
        if self._sa or self._sb:
            self.state = PERM_SAT


class _ExclusiveChoice(_Pair):
    """exclusive_choice(a,b): exactly one occurs.  Pending while neither has
    been seen, violated once both have."""
    __slots__ = ()

    def _update(self, x, prev):
        self._mark(x)
        if self._sa and self._sb:
            self.state = PERM_VIOL
        elif self._sa or self._sb:
            self.state = POSS_SAT


class _NotCoExistence(_Pair):
    """not_coexistence(a,b): they never both occur."""
    __slots__ = ()

    def __init__(self, c):
        super().__init__(c)
        self.state = POSS_SAT

    def _update(self, x, prev):
        self._mark(x)
        if self._sa and self._sb:
            self.state = PERM_VIOL


MONITORS = {
    "init": _Init, "end": _End, "existence": _Existence, "absence": _Absence,
    "exactly": _Exactly, "response": _Response, "precedence": _Precedence,
    "succession": _Succession, "alternate_response": _AlternateResponse,
    "alternate_precedence": _AlternatePrecedence,
    "chain_response": _ChainResponse, "chain_precedence": _ChainPrecedence,
    "chain_succession": _ChainSuccession, "not_succession": _NotSuccession,
    "not_chain_succession": _NotChainSuccession, "coexistence": _CoExistence,
    "choice": _Choice, "exclusive_choice": _ExclusiveChoice,
    "not_coexistence": _NotCoExistence,
}


def monitor_for(c: Constraint) -> Monitor:
    return MONITORS[c.template](c)


# --------------------------------------------------------------------------- #
# Witness trackers -- existential, sticky, and independent of the monitor
# --------------------------------------------------------------------------- #

class Witness:
    """Has this prefix produced ONE instance of the pattern?

    Deliberately separate from Monitor: a witness is a property of the prefix,
    not of the constraint's satisfaction, so it must keep accruing after the
    constraint has been permanently violated.
    """
    __slots__ = ("a", "b", "n", "hit", "_s1", "_s2")

    def __init__(self, c: Constraint):
        self.a, self.b, self.n = c.source, c.target, c.n
        self.hit = False
        self._s1 = 0            # per-template scratch (count / flag)
        self._s2 = 0

    def push(self, x: str, prev: Optional[str], first: bool) -> bool:
        """Returns True on the event that first establishes the witness."""
        if self.hit:
            return False
        self._update(x, prev, first)
        return self.hit

    def _update(self, x, prev, first):
        raise NotImplementedError


class _WNever(Witness):
    __slots__ = ()

    def _update(self, x, prev, first):
        return


class _WInit(Witness):
    __slots__ = ()

    def _update(self, x, prev, first):
        if first and x == self.a:
            self.hit = True


class _WExistence(Witness):
    __slots__ = ()

    def _update(self, x, prev, first):
        if x == self.a:
            self._s1 += 1
            if self._s1 >= (self.n or 1):
                self.hit = True


class _WOrdered(Witness):
    """some `a` before some `b` -- response, precedence, succession."""
    __slots__ = ()

    def _update(self, x, prev, first):
        if x == self.b and self._s1:
            self.hit = True
        if x == self.a:
            self._s1 = 1


class _WAltResponse(Witness):
    """some `a` followed by a `b` with no intervening `a`.

    The early return matters when a == b: an `a` can then never be *strictly*
    between two consecutive `a`s, so the pattern has no witness at all, and
    returning before the `b` branch is what encodes that.
    """
    __slots__ = ()

    def _update(self, x, prev, first):
        if x == self.a:
            self._s1 = 1
            return
        if x == self.b and self._s1:
            self.hit = True


class _WAltPrecedence(Witness):
    """some `b` preceded by an `a` with no intervening `b`.

    Symmetric to the above: credit must NOT be re-established by the very event
    that becomes the previous `b`, or a == b would witness itself.
    """
    __slots__ = ()

    def _update(self, x, prev, first):
        if x == self.b:
            if self._s1:
                self.hit = True
            self._s1 = 0
            return
        if x == self.a:
            self._s1 = 1


class _WBigram(Witness):
    """the bigram (a,b) -- chain_response, chain_precedence, chain_succession."""
    __slots__ = ()

    def _update(self, x, prev, first):
        if prev == self.a and x == self.b:
            self.hit = True


class _WCoExistence(Witness):
    __slots__ = ()

    def _update(self, x, prev, first):
        if x == self.a:
            self._s1 = 1
        if x == self.b:
            self._s2 = 1
        if self._s1 and self._s2:
            self.hit = True


WITNESSES = {
    "init": _WInit, "existence": _WExistence,
    "response": _WOrdered, "precedence": _WOrdered, "succession": _WOrdered,
    "alternate_response": _WAltResponse,
    "alternate_precedence": _WAltPrecedence,
    "chain_response": _WBigram, "chain_precedence": _WBigram,
    "chain_succession": _WBigram, "coexistence": _WCoExistence,
}


def witness_for(c: Constraint) -> Witness:
    return WITNESSES.get(c.template, _WNever)(c)


# --------------------------------------------------------------------------- #
# The batch oracles -- INDEPENDENT implementations, used as ground truth
# --------------------------------------------------------------------------- #

def satisfies(template: str, source: str, target: Optional[str],
              n: Optional[int], labels: Sequence[str]) -> bool:
    """Does the FINISHED trace `labels` satisfy the constraint?

    Written from the index-based definitions in siesta/modules/mine/*.py rather
    than from the monitors above, so that tests/test_declare.py compares two
    genuinely separate derivations.  No vacuity rule is applied here: whether a
    constraint is *mined* on a trace is a question for mine_local().
    """
    L = list(labels)
    a, b = source, target
    src = [i for i, x in enumerate(L) if x == a]
    tgt = [i for i, x in enumerate(L) if x == b] if b is not None else []

    if template == "init":
        return bool(L) and L[0] == a
    if template == "end":
        return bool(L) and L[-1] == a
    if template == "existence":
        return len(src) >= (n or 1)
    if template == "absence":
        return len(src) < (n or 1)
    if template == "exactly":
        return len(src) == (n or 0)

    if template == "response":
        return all(any(j > i for j in tgt) for i in src)
    if template == "precedence":
        return all(any(i < j for i in src) for j in tgt)
    if template == "succession":
        return (satisfies("response", a, b, n, L)
                and satisfies("precedence", a, b, n, L))

    if template == "alternate_response":
        for k, i in enumerate(src):
            later = [j for j in tgt if j > i]
            if not later:
                return False
            if k + 1 < len(src) and later[0] >= src[k + 1]:
                return False
        return True
    if template == "alternate_precedence":
        for k, j in enumerate(tgt):
            earlier = [i for i in src if i < j]
            if not earlier:
                return False
            if k > 0 and earlier[-1] <= tgt[k - 1]:
                return False
        return True

    if template == "chain_response":
        return all(i + 1 < len(L) and L[i + 1] == b for i in src)
    if template == "chain_precedence":
        return all(j - 1 >= 0 and L[j - 1] == a for j in tgt)
    if template == "chain_succession":
        return (satisfies("chain_response", a, b, n, L)
                and satisfies("chain_precedence", a, b, n, L))

    if template == "not_succession":
        return not (src and tgt and min(src) < max(tgt))
    if template == "not_chain_succession":
        return not any(i + 1 < len(L) and L[i + 1] == b for i in src)

    sa, sb = bool(src), bool(tgt)
    if template == "coexistence":
        return sa and sb
    if template == "choice":
        return sa or sb
    if template == "exclusive_choice":
        return sa != sb
    if template == "not_coexistence":
        return not (sa and sb)

    raise ValueError(f"unknown template {template!r}")


def witnesses(template: str, source: str, target: Optional[str],
              n: Optional[int], labels: Sequence[str]) -> bool:
    """Batch form of the existential witness (see the Witness class)."""
    L = list(labels)
    a, b = source, target
    src = [i for i, x in enumerate(L) if x == a]
    tgt = [i for i, x in enumerate(L) if x == b] if b is not None else []

    if template == "init":
        return bool(L) and L[0] == a
    if template == "existence":
        return len(src) >= (n or 1)
    if template in ("response", "precedence", "succession"):
        return any(i < j for i in src for j in tgt)
    if template == "alternate_response":
        for k, i in enumerate(src):
            nxt = src[k + 1] if k + 1 < len(src) else len(L)
            if any(i < j < nxt for j in tgt):
                return True
        return False
    if template == "alternate_precedence":
        for k, j in enumerate(tgt):
            prv = tgt[k - 1] if k > 0 else -1
            if any(prv < i < j for i in src):
                return True
        return False
    if template in ("chain_response", "chain_precedence", "chain_succession"):
        return any(i + 1 < len(L) and L[i + 1] == b for i in src)
    if template == "coexistence":
        return bool(src) and bool(tgt)
    return False


# --------------------------------------------------------------------------- #
# Model + streaming state
# --------------------------------------------------------------------------- #

class _Slot:
    """A constraint's monitor and witness tracker, woken together."""
    __slots__ = ("c", "mon", "wit", "stamp", "idx")

    def __init__(self, c: Constraint, idx: int = 0):
        self.c = c
        self.idx = idx
        self.mon = monitor_for(c)
        self.wit = witness_for(c)
        self.stamp = -1


class DeclareModel:
    """A mined constraint set, indexed so that push() touches only the monitors
    an event can possibly affect."""

    def __init__(self, constraints: Sequence[Constraint]):
        self.constraints: List[Constraint] = list(constraints)
        self.total_weight = sum(c.weight for c in self.constraints) or 1.0

    def __len__(self) -> int:
        return len(self.constraints)

    def monitors(self) -> "DeclareState":
        return DeclareState(self)

    def template_census(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.constraints:
            out[c.template] = out.get(c.template, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


class DeclareState:
    """Streaming evaluation of a whole DeclareModel over one run.

    Counters are maintained incrementally.  A monitor that is not woken cannot
    change state, which is what makes that sound.
    """
    __slots__ = ("model", "slots", "by_label", "after_label", "always",
                 "act_by_label", "act_any", "dirty", "_seen",
                 "n_perm_viol", "w_perm_viol", "n_pending", "n_activated",
                 "n_witness", "w_witness", "n_events", "_prev", "_tick")

    def __init__(self, model: DeclareModel):
        self.model = model
        self.slots = [_Slot(c, i) for i, c in enumerate(model.constraints)]
        self.by_label: Dict[str, List[_Slot]] = {}
        self.after_label: Dict[str, List[_Slot]] = {}
        self.always: List[_Slot] = []
        for s in self.slots:
            labels, after, always = wake_sets(s.c)
            if always:
                self.always.append(s)
            for lab in labels:
                if lab is not None:
                    self.by_label.setdefault(lab, []).append(s)
            for lab in after:
                if lab is not None:
                    self.after_label.setdefault(lab, []).append(s)

        # activation index: first sighting of a label activates every
        # constraint that label is an antecedent of
        self.act_by_label: Dict[str, List[int]] = {}
        self.act_any: List[int] = []
        for s2 in self.slots:
            labs = activation_labels(s2.c)
            if not labs:
                self.act_any.append(s2.idx)
            for lab in labs:
                if lab is not None:
                    self.act_by_label.setdefault(lab, []).append(s2.idx)
        self._seen: set = set()
        self.dirty: List[int] = []

        self.n_perm_viol = 0
        self.w_perm_viol = 0.0
        self.n_pending = sum(1 for s in self.slots if s.mon.state == POSS_VIOL)
        self.n_activated = 0
        self.n_witness = 0
        self.w_witness = 0.0
        self.n_events = 0
        self._prev: Optional[str] = None
        self._tick = 0

    def push(self, lab: str) -> None:
        self._tick += 1
        tick = self._tick
        self.n_events += 1
        first = tick == 1
        prev = self._prev
        self.dirty = []

        # activation first, and centrally: it is a function of the label SET,
        # so it cannot be left to monitors that are not woken by their own
        # antecedent.  Only the FIRST occurrence of a label can activate.
        if lab not in self._seen:
            self._seen.add(lab)
            for idx in self.act_by_label.get(lab, ()):
                m = self.slots[idx].mon
                if not m._act:
                    m._act = True
                    self.n_activated += 1
                    self.dirty.append(idx)
        if first:
            for idx in self.act_any:
                m = self.slots[idx].mon
                if not m._act:
                    m._act = True
                    self.n_activated += 1
                    self.dirty.append(idx)

        groups = (self.always, self.by_label.get(lab),
                  self.after_label.get(prev) if prev is not None else None)
        for group in groups:
            if not group:
                continue
            for s in group:
                if s.stamp == tick:
                    continue                     # already woken this event
                s.stamp = tick
                m = s.mon
                before = m.state
                after = m.push(lab, prev)
                if after != before:
                    if before == POSS_VIOL:
                        self.n_pending -= 1
                    if after == POSS_VIOL:
                        self.n_pending += 1
                    if after == PERM_VIOL:
                        self.n_perm_viol += 1
                        self.w_perm_viol += s.c.weight
                    self.dirty.append(s.idx)
                if s.wit.push(lab, prev, first):
                    self.n_witness += 1
                    self.w_witness += s.c.weight
        self._prev = lab

    # -- read-outs --------------------------------------------------------- #

    @property
    def viol_ratio(self) -> float:
        return self.n_perm_viol / self.n_activated if self.n_activated else 0.0

    @property
    def weighted_viol(self) -> float:
        return self.w_perm_viol / self.model.total_weight

    @property
    def weighted_witness(self) -> float:
        return self.w_witness / self.model.total_weight

    #: The four signature spaces E24 compares runs in.  `activated` is the
    #: user-facing one and is order-BLIND by construction (see ACTIVATED_BY);
    #: the other three read a monitor's verdict and are order-sensitive.
    SIGNATURES = ("activated", "state", "violated", "pending")

    def symbol(self, idx: int, kind: str):
        """This slot's contribution to the signature set, or None."""
        m = self.slots[idx].mon
        if kind == "activated":
            return idx if m._act else None
        if kind == "state":
            return (idx, m.state) if m._act else None
        if kind == "violated":
            return idx if m.state == PERM_VIOL else None
        if kind == "pending":
            return idx if m.state == POSS_VIOL else None
        raise ValueError(f"unknown signature {kind!r}")

    def signature(self, kind: str) -> set:
        out = set()
        for i in range(len(self.slots)):
            sym = self.symbol(i, kind)
            if sym is not None:
                out.add(sym)
        return out

    def states(self) -> Dict[str, int]:
        out = {v: 0 for v in STATE_NAMES.values()}
        for s in self.slots:
            out[STATE_NAMES[s.mon.state]] += 1
        return out


# --------------------------------------------------------------------------- #
# Local miner -- a faithful reimplementation of the SIESTA pipeline
# --------------------------------------------------------------------------- #
#
# WHY THIS EXISTS.  SIESTA is the miner of record, but it mines synchronously
# over Spark with no job id, and E22 needs one model per (corpus, class, seed) --
# 80 of them.  So the sweep runs against this reimplementation, and
# experiments/e22_declare_check.py asserts that the two agree row for row on a
# real log before any of it is believed.
#
# Reproducing it means reproducing its VACUITY RULES, which differ per category
# and are not the textbook ones:
#
#   ordered      pairs are drawn from the activities of THAT TRACE, so support
#                is non-vacuous: a constraint is only counted on traces where
#                both activities occur       (mine/ordered.py:41-45)
#   unordered    pairs are drawn from the GLOBAL activity set, skipping only
#                traces where neither occurs (mine/unordered.py:33-40)
#   negation     global pairs, no skip at all -- support is
#                (N - coexistence count) / N (mine/negations.py:90-97)
#   existential  `existence(a,n)` and `absence(a,n+1)` are RELABELLED
#                `exactly(a,n)` rows, and main.py:289 groups by `occurrences`,
#                so the mined support of existence(a,n) is P(count == n), not
#                the cumulative P(count >= n)  (mine/existential.py:57-61)
#
# The last one is a genuine upstream defect rather than a convention.  It is
# reproduced here so the cross-check is exact; `cumulative_existential=True`
# gives the semantically intended reading for actual use.

def _labels_of(t) -> List[str]:
    lab = getattr(t, "labels", None)
    return list(lab) if lab is not None else list(t)


def _mine_one(labels: List[str], global_acts: Sequence[str],
              templates: frozenset) -> List[Tuple]:
    """Every constraint key the SIESTA miners emit for a single trace."""
    rows: List[Tuple] = []
    if not labels:
        return rows
    add = rows.append
    present = set(labels)

    # -- positional ------------------------------------------------------- #
    if "init" in templates:
        add(("init", labels[0], None, None))
    if "end" in templates:
        add(("end", labels[-1], None, None))

    # -- existential: one exactly/existence/absence row per activity ------- #
    if templates & {"exactly", "existence", "absence"}:
        for a in sorted(present):
            cnt = labels.count(a)
            if "exactly" in templates:
                add(("exactly", a, None, cnt))
            if "existence" in templates:
                add(("existence", a, None, cnt))
            if "absence" in templates:
                add(("absence", a, None, cnt + 1))

    # -- ordered: pairs from THIS trace's activities, both directions ------ #
    acts = sorted(present)
    pos = {a: [i for i, x in enumerate(labels) if x == a] for a in acts}
    for a in acts:
        for b in acts:
            # SIESTA skips every positive template unless some `a` precedes
            # some `b`; its early `break` also forces the alternate variants
            # false, which is consistent because alternate_* implies its base.
            if pos[a][0] < pos[b][-1]:
                for t in ("response", "precedence", "succession",
                          "alternate_response", "alternate_precedence",
                          "chain_response", "chain_precedence",
                          "chain_succession", "not_chain_succession"):
                    if t in templates and satisfies(t, a, b, None, labels):
                        add((t, a, b, None))
            else:
                if "not_succession" in templates:
                    add(("not_succession", a, b, None))
                if ("not_chain_succession" in templates
                        and satisfies("not_chain_succession", a, b, None, labels)):
                    add(("not_chain_succession", a, b, None))

    # -- unordered + negation: GLOBAL pairs, source < target --------------- #
    want_un = templates & {"coexistence", "choice", "exclusive_choice"}
    if want_un or "not_coexistence" in templates:
        for i, a in enumerate(global_acts):
            a_in = a in present
            for b in global_acts[i + 1:]:
                b_in = b in present
                if "not_coexistence" in templates and not (a_in and b_in):
                    add(("not_coexistence", a, b, None))
                if not (a_in or b_in):
                    continue            # unordered skips these, negation did not
                if a_in and b_in and "coexistence" in templates:
                    add(("coexistence", a, b, None))
                if "choice" in templates:
                    add(("choice", a, b, None))
                if a_in != b_in and "exclusive_choice" in templates:
                    add(("exclusive_choice", a, b, None))
    return rows


def mine_local(traces: Sequence, templates: Optional[Sequence[str]] = None,
               support_threshold: float = 0.0, confidence_threshold: float = 0.0,
               interest_threshold: float = 0.0, siesta_compat: bool = False,
               cumulative_existential: bool = False) -> List[Constraint]:
    """Mine DECLARE constraints, reproducing SIESTA's statistics exactly.

    `siesta_compat=True` also reproduces its NULL-comparison filters, under
    which no `positional` or `existential` row can ever reach the output (their
    target is NULL, so interest is NULL, so `interest >= threshold` is NULL and
    Spark drops the row).  That is what makes the cross-check exact; leave it
    off for real use or 5 of the 19 templates silently vanish.

    `cumulative_existential=True` reports existence(a,n) as P(count >= n) and
    absence(a,n) as P(count < n), which is what the templates actually mean.
    """
    want = frozenset(templates) if templates is not None else frozenset(TEMPLATES)
    label_lists = [_labels_of(t) for t in traces]
    n_traces = len(label_lists)
    if n_traces == 0:
        return []

    global_acts = sorted({x for L in label_lists for x in L})
    act_traces: Dict[str, int] = {}
    for L in label_lists:
        for a in set(L):
            act_traces[a] = act_traces.get(a, 0) + 1

    counts: Dict[Tuple, int] = {}
    for L in label_lists:
        for key in _mine_one(L, global_acts, want):
            counts[key] = counts.get(key, 0) + 1

    if cumulative_existential:
        counts = _accumulate_existential(counts, act_traces, n_traces)

    out: List[Constraint] = []
    for (tmpl, a, b, occ), match in sorted(counts.items(), key=lambda kv: str(kv[0])):
        cat = CATEGORY[tmpl]
        support = match / n_traces
        if support < support_threshold:
            continue
        src_cnt = act_traces.get(a, 0)
        tgt_cnt = act_traces.get(b) if b is not None else None

        # confidence, branched exactly as main.py:342-355
        if cat == "unordered":
            denom = (src_cnt * (tgt_cnt or 0)) ** 0.5
            conf = match / denom if denom else None
        elif cat == "negation":
            coex = n_traces - match
            if src_cnt and tgt_cnt:
                conf = (((src_cnt - coex) / src_cnt)
                        * ((tgt_cnt - coex) / tgt_cnt)) ** 0.5
            else:
                conf = None
        else:                                    # ordered / positional / existential
            conf = match / src_cnt if src_cnt else None
        if conf is None or conf < confidence_threshold:
            continue                             # NULL >= x is NULL -> dropped

        # interest = support / expected support under independence
        if src_cnt and tgt_cnt:
            indep = (src_cnt / n_traces) * (tgt_cnt / n_traces)
            expected = (1.0 - indep) if cat == "negation" else indep
            interest = support / expected if expected else None
        else:
            interest = None
        if siesta_compat and (interest is None or interest < interest_threshold):
            continue                             # the NULL-interest defect
        if not siesta_compat and interest is not None and interest < interest_threshold:
            continue

        out.append(Constraint(tmpl, a, b, occ, support, conf, interest))
    return out


def select_discriminative(pos: Sequence[Constraint], neg: Sequence[Constraint],
                          min_delta: float = 0.20, min_support: float = 0.60,
                          top_k: int = 200,
                          exclude: Sequence[str] = DEFAULT_EXCLUDED
                          ) -> Tuple[List[Constraint], List[Constraint]]:
    """Split two mined models into a discriminative positive and negative set.

    Mining successes alone is not enough: `precedence(user, respond)` holds on
    every run of either class and carries nothing.  What matters is the GAP in
    support between the classes, so both models are mined and each constraint is
    scored by delta = support(success) - support(failure).

    Weights are |delta|, not raw support, for the same reason -- weighting by
    support puts the tautologies at the top, which is exactly backwards.

    The negative set is restricted to WITNESS_TEMPLATES: a universal negative
    ("`a` never precedes `b`") cannot be confirmed by any prefix, so a
    constraint mined from failures is only usable online if one instance of it
    is enough to recognise.
    """
    drop = set(exclude)
    P = {c.key: c for c in pos if c.template not in drop}
    N = {c.key: c for c in neg if c.template not in drop}

    plus, minus = [], []
    for key in set(P) | set(N):
        sp = P[key].support if key in P else 0.0
        sf = N[key].support if key in N else 0.0
        delta = sp - sf
        base = P.get(key) or N[key]
        if sp >= min_support and delta >= min_delta:
            plus.append(Constraint(base.template, base.source, base.target,
                                   base.n, sp, base.confidence, base.interest,
                                   weight=abs(delta)))
        elif (sf >= min_support and -delta >= min_delta
                and base.template in WITNESS_TEMPLATES):
            minus.append(Constraint(base.template, base.source, base.target,
                                    base.n, sf, base.confidence, base.interest,
                                    weight=abs(delta)))
    plus.sort(key=lambda c: (-c.weight, str(c.key)))
    minus.sort(key=lambda c: (-c.weight, str(c.key)))
    return plus[:top_k], minus[:top_k]


# --------------------------------------------------------------------------- #
# Streaming scores -- the repo's detector interface: push(ev) -> [0,1]
# --------------------------------------------------------------------------- #

import math                                                        # noqa: E402

from tracebank.events import Event                                 # noqa: E402


def _lab(ev) -> str:
    return ev if isinstance(ev, str) else ev.act


#: FEATURES[name] = (needs_positive, needs_negative, docstring)
FEATURES = {
    "dcl_pos_viol":  "permanently violated positive constraints, squashed",
    "dcl_pos_ratio": "violated / activated -- a rate whose denominator varies",
    "dcl_pos_pend":  "outstanding obligations, squashed",
    "dcl_pos_wviol": "violated positive constraints, weighted by |delta support|",
    "dcl_neg_wit":   "witnessed failure signatures, squashed",
    "dcl_neg_wwit":  "witnessed failure signatures, weighted",
    "dcl_both":      "positive violations and negative witnesses, averaged",
    "dcl_viol_rate": "CONTROL: violations / step -- identical at a fixed step",
}
NEEDS_NEGATIVE = ("dcl_neg_wit", "dcl_neg_wwit", "dcl_both")


class DeclareScorer:
    """One streaming feature over a positive and/or negative DECLARE model.

    Higher is more divergent, as everywhere else in the repo.  The exponential
    squash only exists to land in [0,1] for the shared calibration path; being
    monotone it cannot change any fixed-step ranking, which is the point of the
    `dcl_viol_rate` control.
    """

    def __init__(self, feature: str, pos: Optional[DeclareModel] = None,
                 neg: Optional[DeclareModel] = None, scale: float = 8.0,
                 name: Optional[str] = None):
        if feature not in FEATURES:
            raise ValueError(f"unknown feature {feature!r}")
        if feature in NEEDS_NEGATIVE and neg is None:
            raise ValueError(f"{feature} needs a negative model")
        if feature not in NEEDS_NEGATIVE and pos is None:
            raise ValueError(f"{feature} needs a positive model")
        # `name` lets the same feature be reported twice over different models
        # -- the full model and the ordering-only one
        self.name = name or feature
        self.feature = feature
        self.scale = scale
        self.pos = pos.monitors() if pos is not None else None
        self.neg = neg.monitors() if neg is not None else None
        self.n = 0

    def _squash(self, x: float) -> float:
        return 1.0 - math.exp(-x / self.scale)

    def push(self, ev) -> float:
        lab = _lab(ev)
        self.n += 1
        if self.pos is not None:
            self.pos.push(lab)
        if self.neg is not None:
            self.neg.push(lab)
        f = self.feature
        if f == "dcl_pos_viol":
            return self._squash(self.pos.n_perm_viol)
        if f == "dcl_pos_ratio":
            return self.pos.viol_ratio
        if f == "dcl_pos_pend":
            return self._squash(self.pos.n_pending)
        if f == "dcl_pos_wviol":
            return min(1.0, self.pos.weighted_viol)
        if f == "dcl_neg_wit":
            return self._squash(self.neg.n_witness)
        if f == "dcl_neg_wwit":
            return min(1.0, self.neg.weighted_witness)
        if f == "dcl_both":
            v = self.pos.n_perm_viol / max(1, len(self.pos.model))
            w = self.neg.n_witness / max(1, len(self.neg.model))
            return min(1.0, 0.5 * (v + w))
        # dcl_viol_rate -- the labelled control
        return min(1.0, self.pos.n_perm_viol / self.n)


def _accumulate_existential(counts: Dict[Tuple, int], act_traces: Dict[str, int],
                            n_traces: int) -> Dict[Tuple, int]:
    """Turn SIESTA's per-n existential rows into the cumulative reading.

    THIS MATTERS FOR CORRECTNESS, not tidiness.  The monitor for existence(a,n)
    tests `count >= n`, so if the mined support said P(count == n) the model
    would be selected on a different property from the one it is then scored by.

    Two corrections, and the second is easy to miss:
      * existence(a,n) sums the UPPER tail of the exact-count distribution,
        absence(a,n) the LOWER tail -- its key is already count+1, so a trace
        with stored key k satisfies absence(a,n) exactly when k <= n.
      * traces where `a` never occurs emit no existential row at all, yet they
        satisfy every absence(a,n).  They have to be added back explicitly.
    """
    out = {k: v for k, v in counts.items() if k[0] not in ("existence", "absence")}
    by_act: Dict[Tuple[str, str], Dict[int, int]] = {}
    for (tmpl, a, b, occ), v in counts.items():
        if tmpl in ("existence", "absence"):
            by_act.setdefault((tmpl, a), {})[occ] = v
    for (tmpl, a), dist in by_act.items():
        absent = n_traces - act_traces.get(a, 0)
        for n in sorted(dist):
            if tmpl == "existence":                       # count >= n
                out[(tmpl, a, None, n)] = sum(v for k, v in dist.items() if k >= n)
            else:                                         # count < n
                out[(tmpl, a, None, n)] = (
                    absent + sum(v for k, v in dist.items() if k <= n))
    return out
