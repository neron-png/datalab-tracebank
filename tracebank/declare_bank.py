"""Divergence measured in CONSTRAINT space rather than label space (E24).

The bank score this project settled on (E7) is `jaccard_prefix / max@5`: compare
the SET OF LABELS the running prefix has produced against the set each bank
reference had produced by the same step.  Order-free set membership beat edit
distance outright, which is the finding -- and also the limitation, since a set
of labels says nothing about the order they arrived in.

This module keeps the bank and the Jaccard and changes the ALPHABET.  Instead of
comparing label sets it compares sets of constraint-derived symbols, drawn from
a DECLARE model mined on the trusted runs.  Four signature spaces:

  activated  the constraints whose antecedent has fired.  THE ONE THE IDEA
             STARTS FROM -- and, by ACTIVATED_BY in declare.py, a function of
             the label SET alone: `response(a,b)` activates on an `a` whenever
             it occurs.  So this is a REWEIGHTED label Jaccard, each label
             weighted by how many constraints it is an antecedent of, and it
             carries no order information.  E24 measures how much that
             reweighting is worth, and reports its rank correlation with plain
             jaccard_prefix as the redundancy check.
  state      (constraint, RV-LTL verdict) for every activated constraint.  The
             verdict depends on order, so this one is order-sensitive: two runs
             using identical tools in different orders differ here and are
             identical under `activated`.
  violated   constraints the prefix has permanently violated.
  pending    constraints with an outstanding obligation.

Everything stays incremental.  A signature symbol changes only for constraints
whose monitor was woken or whose antecedent just fired, which DeclareState
reports as `dirty`, so a step costs O(|dirty| * k) rather than O(|C| * k).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from tracebank.declare import DeclareModel, DeclareState

AGGREGATORS = ("max_knn", "min", "mean_knn", "mean")


def _labels_of(t) -> List[str]:
    lab = getattr(t, "labels", None)
    return list(lab) if lab is not None else list(t)


def _lab(ev) -> str:
    return ev if isinstance(ev, str) else ev.act


class SignatureBank:
    """Bank references pre-encoded as constraint-signature sets.

    For `prefix` mode the reference set grows alongside the prefix, matching
    what `IncrementalJaccard(mode="prefix")` does with labels: a reference
    shorter than the prefix simply saturates at its final signature.
    """

    def __init__(self, model: DeclareModel, traces: Sequence, kind: str,
                 mode: str = "prefix"):
        if mode not in ("prefix", "full"):
            raise ValueError(f"unknown mode {mode!r}")
        self.model = model
        self.kind = kind
        self.mode = mode
        #: per reference, the per-step (added, removed) symbol deltas
        # The signature BEFORE any event is not always empty: `exactly`,
        # `existence`, `coexistence`, `choice` and `exclusive_choice` all start
        # in POSS_VIOL, so the `pending` signature is populated at step 0.
        # `dirty` only reports CHANGES, so both sides have to be seeded from
        # this baseline or those symbols never enter the incremental sets.
        self.initial = model.monitors().signature(kind)

        self.deltas: List[List] = []
        self.finals: List[set] = []
        for t in traces:
            st = model.monitors()
            cur: Dict[int, object] = {i: st.symbol(i, kind)
                                      for i in range(len(st.slots))}
            sig: set = set(self.initial)
            steps = []
            for lab in _labels_of(t):
                st.push(lab)
                added, removed = [], []
                for idx in set(st.dirty):
                    new = st.symbol(idx, kind)
                    old = cur.get(idx)
                    if new == old:
                        continue
                    if old is not None:
                        removed.append(old)
                        sig.discard(old)
                    if new is not None:
                        added.append(new)
                        sig.add(new)
                    cur[idx] = new
                steps.append((added, removed))
            self.deltas.append(steps)
            self.finals.append(set(sig))

    def __len__(self) -> int:
        return len(self.deltas)


class DeclareBankScorer:
    """delta(sigma^(i), B) computed as a Jaccard in constraint space.

    Same shape as SessionScorer: push(event) -> [0,1], higher is more divergent,
    and the same aggregators, including the identity that "the largest of the
    kappa nearest" is just the kappa-th smallest.
    """

    def __init__(self, sigbank: SignatureBank, kappa: int = 5,
                 aggregator: str = "max_knn", name: Optional[str] = None):
        if aggregator not in AGGREGATORS:
            raise ValueError(f"unknown aggregator {aggregator!r}")
        self.sb = sigbank
        self.kappa = kappa
        self.aggregator = aggregator
        self.name = name or f"dcl_jac_{sigbank.kind}"
        self.st = sigbank.model.monitors()
        self.k = len(sigbank)
        # seed both sides from the step-0 signature (see SignatureBank.initial)
        self._cur: Dict[int, object] = {
            i: self.st.symbol(i, sigbank.kind) for i in range(len(self.st.slots))}
        self._p: set = set(sigbank.initial)
        self._r: List[set] = [set(sigbank.initial) for _ in range(self.k)]
        self._inter: List[int] = [len(sigbank.initial)] * self.k
        self._n = 0

    def push(self, ev) -> float:
        self.st.push(_lab(ev))
        self._n += 1

        # --- prefix side: apply only the symbols that actually changed ----- #
        for idx in set(self.st.dirty):
            new = self.st.symbol(idx, self.sb.kind)
            old = self._cur.get(idx)
            if new == old:
                continue
            if old is not None:
                self._p.discard(old)
                for j in range(self.k):
                    if old in self._r[j]:
                        self._inter[j] -= 1
            if new is not None:
                self._p.add(new)
                for j in range(self.k):
                    if new in self._r[j]:
                        self._inter[j] += 1
            self._cur[idx] = new

        # --- reference side ------------------------------------------------ #
        # Order matters: the prefix is updated first, so a symbol entering both
        # sets on the same step is counted exactly once (it misses on the
        # prefix pass and hits on the reference pass).
        if self.sb.mode == "prefix":
            for j, steps in enumerate(self.sb.deltas):
                if self._n > len(steps):
                    continue                      # reference exhausted, saturate
                added, removed = steps[self._n - 1]
                rj = self._r[j]
                for sym in removed:
                    if sym in rj:
                        rj.discard(sym)
                        if sym in self._p:
                            self._inter[j] -= 1
                for sym in added:
                    if sym not in rj:
                        rj.add(sym)
                        if sym in self._p:
                            self._inter[j] += 1
        elif self._n == 1:
            for j in range(self.k):
                self._r[j] = set(self.sb.finals[j])
                self._inter[j] = len(self._p & self._r[j])

        dists = []
        np_ = len(self._p)
        for j in range(self.k):
            inter = self._inter[j]
            union = np_ + len(self._r[j]) - inter
            dists.append(1.0 - inter / union if union else 0.0)
        return self._reduce(dists)

    def _reduce(self, d: List[float]) -> float:
        if not d:
            return 0.0
        d = sorted(d)
        if self.aggregator == "min":
            return d[0]
        if self.aggregator == "max_knn":
            # the largest of the kappa nearest IS the kappa-th smallest
            return d[min(self.kappa, len(d)) - 1]
        if self.aggregator == "mean_knn":
            m = d[:min(self.kappa, len(d))]
            return sum(m) / len(m)
        return sum(d) / len(d)


def batch_signature(model: DeclareModel, labels: Sequence[str], kind: str,
                    upto: Optional[int] = None) -> set:
    """The signature of a prefix, computed from scratch.  Reference definition
    for tests/test_declare.py -- the incremental path above must equal it."""
    st = model.monitors()
    for lab in list(labels)[:upto]:
        st.push(lab)
    return st.signature(kind)
