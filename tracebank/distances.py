"""Incremental prefix-to-reference distances (Section 4.2).

Every scorer below is *streaming*: push(label) appends one event and returns the
updated normalised distance in [0,1].  Nothing about the prefix is stored beyond
a constant number of DP rows / counters, so cost per step is O(m) for edit
distance and O(1) for the set measures, with m the reference length.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

# --------------------------------------------------------------------------- #
# Damerau-Levenshtein (optimal string alignment variant)
# --------------------------------------------------------------------------- #

NORMALISERS = ("free_suffix", "full", "truncated")


class IncrementalDL:
    """Damerau-Levenshtein (OSA) between a growing prefix and a fixed reference.

    Rows index the prefix, columns the reference.  Appending one event adds one
    row, computed from the two rows already held -- the two preceding rows are
    what the transposition rule needs.  Memory is 3 rows of length m+1.

    normaliser:
      free_suffix : min_j D[n][j] / n   -- prefix alignment; the reference is
                    allowed to be unfinished, which is what we want online.
                    0 means the prefix is exactly a prefix of the reference.
      full        : D[n][m] / max(n, m) -- whole-trace distance.
      truncated   : D[n][j*] / max(n, j*), j* = min(n, m) -- compare against the
                    reference as it looked after the same number of steps.
    """

    __slots__ = ("ref", "m", "n", "_r0", "_r1", "_r2", "normaliser", "_last")

    def __init__(self, reference: Sequence[str], normaliser: str = "free_suffix"):
        if normaliser not in NORMALISERS:
            raise ValueError(f"unknown normaliser {normaliser!r}")
        self.ref = list(reference)
        self.m = len(self.ref)
        self.normaliser = normaliser
        self.n = 0
        self._r2: List[int] = []                       # D[n-2]
        self._r1: List[int] = []                       # D[n-1]
        self._r0: List[int] = list(range(self.m + 1))  # D[n]  (n = 0)
        self._last: str = ""

    # -- streaming ---------------------------------------------------------- #
    def push(self, label: str) -> float:
        ref, m = self.ref, self.m
        prev, prev2 = self._r0, self._r1
        self.n += 1
        n = self.n
        cur = [0] * (m + 1)
        cur[0] = n
        for j in range(1, m + 1):
            cost = 0 if label == ref[j - 1] else 1
            v = min(prev[j] + 1,          # deletion from prefix
                    cur[j - 1] + 1,       # insertion of reference symbol
                    prev[j - 1] + cost)   # match / substitution
            # transposition: prefix[n-2:n] == reversed(ref[j-2:j])
            if n > 1 and j > 1 and label == ref[j - 2] and self._last == ref[j - 1]:
                t = prev2[j - 2] + 1
                if t < v:
                    v = t
            cur[j] = v
        self._r2, self._r1, self._r0 = self._r1, prev, cur
        self._last = label
        return self.score()

    # -- readout ------------------------------------------------------------ #
    def raw(self) -> int:
        if self.n == 0:
            return 0
        if self.normaliser == "free_suffix":
            return min(self._r0)
        if self.normaliser == "full":
            return self._r0[self.m]
        return self._r0[min(self.n, self.m)]

    def score(self) -> float:
        n = self.n
        if n == 0:
            return 0.0
        raw = self.raw()
        if self.normaliser == "free_suffix":
            denom = n
        elif self.normaliser == "full":
            denom = max(n, self.m)
        else:
            denom = max(n, min(n, self.m))
        return min(1.0, raw / denom) if denom else 0.0


def dl_distance(a: Sequence[str], b: Sequence[str], normaliser: str = "free_suffix") -> float:
    """Batch reference implementation (used to verify the incremental one)."""
    d = IncrementalDL(b, normaliser=normaliser)
    for x in a:
        d.push(x)
    return d.score()


# --------------------------------------------------------------------------- #
# Set-based measures
# --------------------------------------------------------------------------- #

class IncrementalJaccard:
    """1 - |P n R| / |P u R| over the *sets* of distinct labels seen so far.

    O(1) per pushed event: a membership test plus two counters.

    mode:
      full       : R is the reference's complete label set.
      prefix     : R grows with the prefix, R = set(ref[:n]) -- like-for-like.
      containment: 1 - |P n R| / |P|, the share of what the agent has done that
                   the reference never does.  Asymmetric, punishes novelty only.
    """

    __slots__ = ("ref", "ref_set", "seen", "inter", "n", "mode", "_rgrow")

    def __init__(self, reference: Sequence[str], mode: str = "full"):
        if mode not in ("full", "prefix", "containment"):
            raise ValueError(f"unknown mode {mode!r}")
        self.ref = list(reference)
        self.mode = mode
        self.seen: set = set()
        self.n = 0
        self.inter = 0
        self._rgrow: set = set()
        self.ref_set = set(self.ref) if mode != "prefix" else self._rgrow

    def push(self, label: str) -> float:
        self.n += 1
        if self.mode == "prefix" and self.n <= len(self.ref):
            r = self.ref[self.n - 1]
            if r not in self._rgrow:
                self._rgrow.add(r)
                if r in self.seen:
                    self.inter += 1
        if label not in self.seen:
            self.seen.add(label)
            if label in self.ref_set:
                self.inter += 1
        return self.score()

    def score(self) -> float:
        if not self.seen:
            return 0.0
        if self.mode == "containment":
            return 1.0 - self.inter / len(self.seen)
        union = len(self.seen) + len(self.ref_set) - self.inter
        return 1.0 - self.inter / union if union else 0.0


class IncrementalNGramJaccard:
    """1 - |P n R| / |P u R| over the SETS of contiguous n-grams.

    The middle ground between order-free Jaccard (n = 1) and full edit distance:
    order matters, but only locally, and the cost stays O(1) per pushed event --
    one new n-gram per step.

    mode:
      full   : R is every n-gram of the whole reference.
      prefix : R = n-grams of ref[:i] as the prefix grows (like-for-like).
    """

    __slots__ = ("ref", "n", "mode", "seen", "ref_set", "inter", "_win",
                 "_rwin", "_ri")

    def __init__(self, reference: Sequence[str], n: int = 2, mode: str = "full"):
        if mode not in ("full", "prefix"):
            raise ValueError(f"unknown mode {mode!r}")
        self.ref = list(reference)
        self.n = n
        self.mode = mode
        self.seen: set = set()
        self.inter = 0
        self._win: List[str] = []
        self._rwin: List[str] = []
        self._ri = 0
        if mode == "full":
            self.ref_set = {tuple(self.ref[i:i + n])
                            for i in range(len(self.ref) - n + 1)}
        else:
            self.ref_set = set()

    def push(self, label: str) -> float:
        n = self.n
        if self.mode == "prefix" and self._ri < len(self.ref):
            self._rwin.append(self.ref[self._ri])
            self._ri += 1
            if len(self._rwin) >= n:
                g = tuple(self._rwin[-n:])
                if g not in self.ref_set:
                    self.ref_set.add(g)
                    if g in self.seen:
                        self.inter += 1
        self._win.append(label)
        if len(self._win) >= n:
            g = tuple(self._win[-n:])
            if g not in self.seen:
                self.seen.add(g)
                if g in self.ref_set:
                    self.inter += 1
        return self.score()

    def score(self) -> float:
        if not self.seen:
            return 0.0
        union = len(self.seen) + len(self.ref_set) - self.inter
        return 1.0 - self.inter / union if union else 0.0


class IncrementalCosine:
    """1 - cos(count(P), count(R)) over label-COUNT vectors.

    Unlike set Jaccard this sees multiplicity, so a prefix that calls one tool
    ten times while the reference calls it once scores as divergent.  O(1) per
    event: update one dot-product term and the prefix norm.
    """

    __slots__ = ("rc", "r_norm", "pc", "dot", "p_sq")

    def __init__(self, reference: Sequence[str]):
        self.rc: Dict[str, int] = {}
        for x in reference:
            self.rc[x] = self.rc.get(x, 0) + 1
        self.r_norm = sum(v * v for v in self.rc.values()) ** 0.5
        self.pc: Dict[str, int] = {}
        self.dot = 0.0
        self.p_sq = 0

    def push(self, label: str) -> float:
        old = self.pc.get(label, 0)
        self.pc[label] = old + 1
        self.p_sq += 2 * old + 1
        self.dot += self.rc.get(label, 0)
        return self.score()

    def score(self) -> float:
        if self.p_sq == 0 or self.r_norm == 0:
            return 0.0 if self.p_sq == 0 else 1.0
        cos = self.dot / ((self.p_sq ** 0.5) * self.r_norm)
        return max(0.0, min(1.0, 1.0 - cos))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

DISTANCES = {
    "dl": lambda ref, **kw: IncrementalDL(ref, normaliser=kw.get("normaliser", "free_suffix")),
    "dl_full": lambda ref, **kw: IncrementalDL(ref, normaliser="full"),
    "dl_trunc": lambda ref, **kw: IncrementalDL(ref, normaliser="truncated"),
    "jaccard": lambda ref, **kw: IncrementalJaccard(ref, mode="full"),
    "jaccard_prefix": lambda ref, **kw: IncrementalJaccard(ref, mode="prefix"),
    "containment": lambda ref, **kw: IncrementalJaccard(ref, mode="containment"),
    "ngram2_jaccard": lambda ref, **kw: IncrementalNGramJaccard(ref, n=2, mode="full"),
    "ngram3_jaccard": lambda ref, **kw: IncrementalNGramJaccard(ref, n=3, mode="full"),
    "ngram2_jaccard_prefix": lambda ref, **kw: IncrementalNGramJaccard(ref, n=2, mode="prefix"),
    "cosine_count": lambda ref, **kw: IncrementalCosine(ref),
}


def make_distance(name: str, reference: Sequence[str], **kw):
    if name not in DISTANCES:
        raise ValueError(f"unknown distance {name!r}; have {sorted(DISTANCES)}")
    return DISTANCES[name](reference, **kw)
