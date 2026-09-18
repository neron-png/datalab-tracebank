"""Topology of the run's walk, rather than the string it spells (E20).

A trace is a walk on a graph whose nodes are activity labels.  Everything else
in this repo compares that walk to reference walks *as a string*.  These scores
instead measure its shape: does it come back to states it had left, does it keep
burning new edges, is it as far along as a reference walk would be by now.

The distinction that matters empirically is RETURN vs ADJACENCY.  Repeating an
action twice in a row (`adj_repeat`, and the existing `repeat_rate` baseline)
carries almost nothing -- AUC 0.49-0.57 at a fixed step on all four corpora.
Re-entering a state you had already *left* carries a lot, and with a sign that
flips by domain:

    fixed-step AUC of return_rate    gpt-4o/retail  0.63   (failures CIRCLE)
                                     both airlines  0.39-0.45 (failures WANDER)

That sign flip is why the two-sided variants exist.  Note that the one-sided
residual CANNOT fix it: subtracting the reference mean at step i is a monotone
transform within a step, so it cannot reorder two runs compared at the same
step, and its fixed-step AUC is identical to the raw feature by construction
(the same trap as delta_resid / delta_stepq in E16).  It is kept as a control
that demonstrates exactly that.  |z| is V-shaped and *can* reorder.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from tracebank.bank import TraceBank


def _label(ev) -> str:
    return ev if isinstance(ev, str) else ev.act


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


# --------------------------------------------------------------------------- #
# raw topology features (no bank)
# --------------------------------------------------------------------------- #

class ReturnRate:
    """Share of steps that re-enter a label the walk had already LEFT.

    Not the same as repeating an action: `a b a` returns, `a a b` does not.
    Streaming, O(1) per step.
    """
    name = "return_rate"

    def __init__(self):
        self.left: set = set()
        self.prev: Optional[str] = None
        self.n = 0
        self.ret = 0

    def push(self, ev) -> float:
        x = _label(ev)
        self.n += 1
        if x in self.left:
            self.ret += 1
        if self.prev is not None and self.prev != x:
            self.left.add(self.prev)
        self.prev = x
        return self.ret / self.n


class EdgeNovelty:
    """Share of transitions that are edges the walk has not taken before.

    High = the run keeps striking out into new territory (the airline failure
    mode); low = it is going round a small subgraph (the retail failure mode).
    """
    name = "edge_novelty"

    def __init__(self):
        self.edges: set = set()
        self.prev: Optional[str] = None
        self.n = 0

    def push(self, ev) -> float:
        x = _label(ev)
        if self.prev is not None:
            self.n += 1
            self.edges.add((self.prev, x))
        self.prev = x
        return len(self.edges) / self.n if self.n else 0.0


# --------------------------------------------------------------------------- #
# per-step reference statistics, taken from trusted runs (no failure labels)
# --------------------------------------------------------------------------- #

def step_profile(refs: Sequence, factory) -> Dict[int, tuple]:
    """(mean, sd) of a streaming feature at each step index, over reference runs."""
    per: Dict[int, List[float]] = defaultdict(list)
    for t in refs:
        f = factory()
        for i, e in enumerate(t.events, 1):
            per[i].append(f.push(e))
    out = {}
    for i, v in per.items():
        m = sum(v) / len(v)
        var = sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)
        out[i] = (m, max(math.sqrt(var), 1e-6))
    return out


class _Referenced:
    """Base: run a raw feature and compare it to the reference profile at step i."""
    factory = None

    def __init__(self, refs: Sequence, squash: float = 1.0):
        self.prof = step_profile(refs, type(self).factory)
        self.max_i = max(self.prof) if self.prof else 0
        self.f = type(self).factory()
        self.n = 0
        self.squash = squash

    def _ref(self):
        i = min(self.n, self.max_i) if self.max_i else self.n
        return self.prof.get(i, (0.0, 1.0))

    def push(self, ev) -> float:
        v = self.f.push(ev)
        self.n += 1
        mu, sd = self._ref()
        return self.combine(v, mu, sd)

    def combine(self, v, mu, sd) -> float:      # pragma: no cover - interface
        raise NotImplementedError


class CycleResidual(_Referenced):
    """return_rate minus what trusted runs show at this step.

    CONTROL, not a candidate: monotone within a step, so its fixed-step AUC is
    identical to plain return_rate and it cannot repair the domain sign flip.
    """
    name = "cycle_resid"
    factory = ReturnRate

    def combine(self, v, mu, sd):
        return _sigmoid((v - mu) / self.squash)


class CycleAbsZ(_Referenced):
    """|z| of return_rate against trusted runs at this step.

    Two-sided on purpose: "an abnormal amount of revisiting, in EITHER
    direction".  V-shaped, so unlike the residual it can reorder runs at a
    fixed step, which is what a domain-dependent sign requires.
    """
    name = "cycle_absz"
    factory = ReturnRate

    def combine(self, v, mu, sd):
        return _sigmoid(abs(v - mu) / sd / self.squash - 1.0)


class EdgeAbsZ(_Referenced):
    """|z| of edge_novelty against trusted runs at this step."""
    name = "edge_absz"
    factory = EdgeNovelty

    def combine(self, v, mu, sd):
        return _sigmoid(abs(v - mu) / sd / self.squash - 1.0)


# --------------------------------------------------------------------------- #
# progress through the graph
# --------------------------------------------------------------------------- #

class GraphProgress:
    """Steps taken, minus where a reference walk would be when it is HERE.

    From the bank, learn the typical position of each label.  If you are at
    `book_reservation` on step 20 and references reach `book_reservation` around
    step 10, you have spent ten steps getting nowhere.  Unlike step_count this
    knows that a long booking is fine and a long lookup is not; and because
    different runs sit at different labels, it still varies at a fixed step.
    """
    name = "graph_progress"

    def __init__(self, bank: TraceBank, scale: float = 5.0):
        pos: Dict[str, List[int]] = defaultdict(list)
        for e in bank.entries:
            for i, lab in enumerate(e.labels, 1):
                pos[lab].append(i)
        self.mu = {k: sum(v) / len(v) for k, v in pos.items()}
        self.scale = scale
        self.n = 0

    def push(self, ev) -> float:
        x = _label(ev)
        self.n += 1
        expected = self.mu.get(x)
        if expected is None:            # a label the bank never uses
            return 0.5
        return _sigmoid((self.n - expected) / self.scale)
