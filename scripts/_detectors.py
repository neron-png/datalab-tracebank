"""One uniform interface over TraceBank scores and the baselines, so that
calibration and the decision-time evaluation share a single code path."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tracebank.bank import ScoreConfig, SessionScorer, TraceBank
from tracebank.baselines import (EmptyRate, ErrorCount, ErrorRate, NGramLM,
                                 PolicyErrorCount, RandomScore, RepeatRate,
                                 StepCount)
from tracebank.events import Trace

Detector = Callable[[], object]     # () -> object with .push(Event) -> float


def tracebank_detector(bank: TraceBank, cfg: ScoreConfig) -> Detector:
    return lambda: SessionScorer(bank, cfg)


def baseline_detectors(bank: TraceBank, seed: int = 0) -> Dict[str, Detector]:
    lm1 = NGramLM(order=1).fit(bank)
    lm0 = NGramLM(order=0).fit(bank)
    return {
        "step_count": lambda: StepCount(),
        "error_rate": lambda: ErrorRate(),
        "error_count": lambda: ErrorCount(),
        "repeat_rate": lambda: RepeatRate(),
        "empty_rate": lambda: EmptyRate(),
        "policy_count": lambda: PolicyErrorCount(),
        "ngram1": lambda: _clone_lm(lm1),
        "ngram0": lambda: _clone_lm(lm0),
        "random": _random_factory(seed),
    }


def _clone_lm(src: NGramLM) -> NGramLM:
    lm = NGramLM(order=src.order, alpha=src.alpha, scale=src.scale)
    lm.vocab, lm.ctx = src.vocab, src.ctx      # shared, read-only
    return lm


def _random_factory(seed: int) -> Detector:
    state = {"i": 0}

    def f():
        state["i"] += 1
        return RandomScore(seed=seed * 100003 + state["i"])
    return f


def stream(det: Detector, trace: Trace) -> List[float]:
    """Score after every event of a completed run."""
    s = det()
    return [s.push(e) for e in trace.events]


# --------------------------------------------------------------------------- #
# Calibration + decision-time replay (identical for every detector)
# --------------------------------------------------------------------------- #

def calibrate(det: Detector, calib: Sequence[Trace], warmup: int = 3,
              target_fpr: float = 0.10) -> float:
    """Threshold = (1 - target_fpr) quantile of the peak score on runs the
    administrator already trusts. Uses no failure labels."""
    peaks = []
    for t in calib:
        if not t.success or len(t) < warmup:
            continue
        sc = stream(det, t)[warmup - 1:]
        if sc:
            peaks.append(max(sc))
    if not peaks:
        return float("inf")
    peaks.sort()
    idx = min(len(peaks) - 1, int(math.ceil((1 - target_fpr) * (len(peaks) - 1))))
    return peaks[idx]


@dataclass
class Decision:
    fired: bool
    step: Optional[int]         # 1-based event index at which it fired
    frac: Optional[float]       # step / run length
    peak: float


def decide(scores: Sequence[float], n: int, threshold: float,
           warmup: int = 3, patience: int = 2) -> Decision:
    streak = 0
    peak = max(scores[warmup - 1:], default=0.0) if len(scores) >= warmup else 0.0
    for i, v in enumerate(scores, 1):
        if i < warmup:
            continue
        # strictly greater: a detector whose calibration quantile sits on a tie
        # (e.g. an error rate of 0.0 on most good runs) must not flag everything
        streak = streak + 1 if v > threshold else 0
        if streak >= patience:
            return Decision(True, i, i / max(1, n), peak)
    return Decision(False, None, None, peak)
