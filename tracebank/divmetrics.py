"""Bank-level divergence scores that are not a per-reference distance.

Each shares the SessionScorer / baseline interface -- push(Event|str) -> [0,1],
higher = more divergent -- so calibration and the decision-time replay treat
them exactly like delta.  Explored in experiment E18.

  DFLegality    fraction of the prefix's directly-follows transitions that occur
                in NO bank trace.  Order-1 conformance, O(1) per step, and a
                rate, so length carries much less into it than into delta.
  DFGFitness    incremental token-replay against the directly-follows graph
                mined from the bank (labels as places, observed transitions as
                arcs, a shared start place).  Score = missing tokens / consumed
                -- the classic replay-fitness cost, streamed.
  StepSurprisal bigram surprisal of the prefix, z-scored against the surprisal
                that TRUSTED runs have accumulated by the same step.  A score
                that removes "a longer run is simply more surprising in total"
                by construction rather than with a threshold curve after it.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence

from tracebank.baselines import NGramLM
from tracebank.bank import TraceBank
from tracebank.events import Event

START = "<start>"


def _label(ev) -> str:
    return ev if isinstance(ev, str) else ev.act


class DFLegality:
    """Share of prefix transitions (prev -> cur) unseen anywhere in the bank."""
    name = "df_legality"

    def __init__(self, bank: TraceBank):
        self.trans: set = set()
        for e in bank.entries:
            L = e.labels
            for a, b in zip(L, L[1:]):
                self.trans.add((a, b))
        self._prev = None
        self.total = 0
        self.illegal = 0

    def push(self, ev) -> float:
        lab = _label(ev)
        if self._prev is not None:
            self.total += 1
            if (self._prev, lab) not in self.trans:
                self.illegal += 1
        self._prev = lab
        return self.illegal / self.total if self.total else 0.0


class DFGFitness:
    """Token-replay cost against the directly-follows graph mined from the bank."""
    name = "dfg_fitness"

    def __init__(self, bank: TraceBank):
        self.pred: Dict[str, set] = defaultdict(set)
        self.starts: set = set()
        for e in bank.entries:
            L = e.labels
            if L:
                self.starts.add(L[0])
            for a, b in zip(L, L[1:]):
                self.pred[b].add(a)
        self.marking: Dict[str, int] = defaultdict(int)
        self.marking[START] = 1
        self.consumed = 0
        self.missing = 0

    def push(self, ev) -> float:
        lab = _label(ev)
        srcs = set(self.pred.get(lab, ()))
        if lab in self.starts:
            srcs.add(START)
        self.consumed += 1
        fired = False
        for s in srcs:
            if self.marking[s] > 0:
                self.marking[s] -= 1
                fired = True
                break
        if not fired:
            self.missing += 1
        self.marking[lab] += 1
        return self.missing / self.consumed if self.consumed else 0.0


class StepSurprisal:
    """Bigram surprisal of the prefix, z-scored against trusted runs at step i.

    The language model is fit on the bank (identical to the `ngram1` baseline).
    The per-step mean / sd of *cumulative* surprisal come from `ref_traces` --
    trusted successful runs held out of the bank -- so no failure labels are
    used and the reference distribution is not the 8 bank traces alone.
    """
    name = "step_surprisal"

    def __init__(self, bank: TraceBank, ref_traces: Sequence,
                 alpha: float = 0.5, squash: float = 2.0):
        self.lm = NGramLM(order=1, alpha=alpha).fit(bank)
        self.squash = squash
        per_step: Dict[int, List[float]] = defaultdict(list)
        for t in ref_traces:
            prev, tot = None, 0.0
            for i, e in enumerate(t.events, 1):
                tot += -self.lm._logp(((prev or "<s>"),), e.act)
                prev = e.act
                per_step[i].append(tot)
        self.mu: Dict[int, float] = {}
        self.sd: Dict[int, float] = {}
        for i, v in per_step.items():
            m = sum(v) / len(v)
            var = sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)
            self.mu[i] = m
            self.sd[i] = max(math.sqrt(var), 1e-6)
        self._max_i = max(per_step) if per_step else 0
        self._prev = None
        self.tot = 0.0
        self.n = 0

    def push(self, ev) -> float:
        lab = _label(ev)
        self.tot += -self.lm._logp(((self._prev or "<s>"),), lab)
        self._prev = lab
        self.n += 1
        i = min(self.n, self._max_i) if self._max_i else self.n
        mu = self.mu.get(i, self.tot)
        sd = self.sd.get(i, 1.0)
        z = max(-30.0, min(30.0, (self.tot - mu) / sd / self.squash))
        return 1.0 / (1.0 + math.exp(-z))
