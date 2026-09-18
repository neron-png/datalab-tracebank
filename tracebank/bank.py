"""The trace bank B and the divergence score delta(sigma^(i), B)  (Section 4.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tracebank.distances import make_distance
from tracebank.events import Event, Trace


@dataclass
class BankEntry:
    labels: List[str]
    run_id: str
    reward: float = 1.0
    task_id: Optional[int] = None
    instruction: Optional[str] = None
    admitted_at: int = 0            # run index at which the judge admitted it
    hits: int = 0                   # times it was a nearest neighbour
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_trace(t: Trace, admitted_at: int = 0) -> "BankEntry":
        return BankEntry(labels=t.labels, run_id=t.run_id, reward=t.reward,
                         task_id=t.task_id, instruction=t.instruction,
                         admitted_at=admitted_at)


class TraceBank:
    """A small set of reference traces, kept as label sequences."""

    def __init__(self, entries: Iterable[BankEntry] = ()):
        self.entries: List[BankEntry] = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @staticmethod
    def from_traces(traces: Sequence[Trace]) -> "TraceBank":
        return TraceBank(BankEntry.from_trace(t) for t in traces)

    def add(self, entry: BankEntry) -> None:
        self.entries.append(entry)

    def remove(self, run_id: str) -> bool:
        n = len(self.entries)
        self.entries = [e for e in self.entries if e.run_id != run_id]
        return len(self.entries) < n

    @property
    def mean_length(self) -> float:
        return sum(len(e.labels) for e in self.entries) / max(1, len(self.entries))


# --------------------------------------------------------------------------- #
# Aggregators over the per-reference distance vector
# --------------------------------------------------------------------------- #

def aggregate(dists: Sequence[float], how: str = "max_knn",
              kappa: int = 3, q: float = 0.25) -> float:
    """Reduce the k per-reference distances to one divergence score.

    max_knn  Eq. (1): max over the kappa nearest references.  Because the kappa
             nearest are the kappa smallest distances, this is exactly the
             kappa-th smallest -- divergent only when the prefix is far from a
             whole cluster of references, not merely from one.
    min      nearest-neighbour search (the kappa = 1 case).
    mean_knn mean of the kappa smallest: smoother, but one close match can mask
             deviation elsewhere.
    quantile the q-quantile of the full distance vector; no kappa to fix, and it
             degrades more gracefully as the bank grows heterogeneous.
    mean     mean over the whole bank.
    """
    if not dists:
        return 0.0
    n = len(dists)
    if how == "min":
        return min(dists)
    if how == "mean":
        return sum(dists) / n
    if how == "max_knn":
        k = max(1, min(kappa, n))
        return sorted(dists)[k - 1]
    if how == "mean_knn":
        k = max(1, min(kappa, n))
        s = sorted(dists)[:k]
        return sum(s) / len(s)
    if how == "quantile":
        s = sorted(dists)
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return s[lo] * (1 - frac) + s[hi] * frac
    raise ValueError(f"unknown aggregator {how!r}")


AGGREGATORS = ("max_knn", "min", "mean_knn", "quantile", "mean")


# --------------------------------------------------------------------------- #
# Per-session streaming scorer
# --------------------------------------------------------------------------- #

@dataclass
class ScoreConfig:
    distance: str = "dl"            # dl | dl_full | dl_trunc | jaccard | ...
    aggregator: str = "max_knn"     # max_knn | min | mean_knn | quantile | mean
    kappa: int = 3
    q: float = 0.25
    name: Optional[str] = None

    def label(self) -> str:
        if self.name:
            return self.name
        tail = {"max_knn": f"@{self.kappa}", "mean_knn": f"@{self.kappa}",
                "quantile": f"@{self.q:g}"}.get(self.aggregator, "")
        return f"{self.distance}/{self.aggregator}{tail}"


class SessionScorer:
    """One live agent session scored against the whole bank.

    Holds one incremental distance object per reference.  push() is
    O(k * m) for edit distance and O(k) for the set measures.
    """

    def __init__(self, bank: TraceBank, cfg: ScoreConfig = ScoreConfig()):
        self.bank = bank
        self.cfg = cfg
        self._d = [make_distance(cfg.distance, e.labels) for e in bank.entries]
        self.n = 0
        self.last: float = 0.0
        self.history: List[float] = []

    def push(self, ev: Event | str) -> float:
        label = ev if isinstance(ev, str) else ev.act
        self.n += 1
        dists = [d.push(label) for d in self._d]
        self.last = aggregate(dists, self.cfg.aggregator, self.cfg.kappa, self.cfg.q)
        self.history.append(self.last)
        return self.last

    def per_reference(self) -> List[float]:
        return [d.score() for d in self._d]

    def nearest(self, k: int = 3) -> List[BankEntry]:
        pairs = sorted(zip(self.per_reference(), range(len(self._d))))
        return [self.bank.entries[i] for _, i in pairs[:k]]


def score_trace(bank: TraceBank, trace: Trace, cfg: ScoreConfig = ScoreConfig()) -> List[float]:
    """Divergence after every step of a completed trace (offline replay)."""
    s = SessionScorer(bank, cfg)
    return [s.push(e) for e in trace.events]
