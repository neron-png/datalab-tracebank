"""Baseline online risk scores to compare delta(sigma^(i), B) against (Section 5.2).

All of them share the SessionScorer interface: push(event) -> score in [0,1],
higher meaning "more likely to be going wrong".
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from tracebank.bank import TraceBank
from tracebank.events import Event


class BaseStream:
    name = "base"

    def push(self, ev: Event) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class StepCount(BaseStream):
    """How long the run has been going, normalised by the step budget.

    The baseline to beat: on tau-bench, failing runs are simply longer.
    """
    name = "step_count"

    def __init__(self, cap: int = 60):
        self.cap = cap
        self.n = 0

    def push(self, ev: Event) -> float:
        self.n += 1
        return min(1.0, self.n / self.cap)


class ErrorRate(BaseStream):
    """Share of tool calls so far that came back `Error: ...`."""
    name = "error_rate"

    def __init__(self):
        self.tools = 0
        self.errs = 0

    def push(self, ev: Event) -> float:
        if ev.attr.get("kind") == "tool":
            self.tools += 1
            if ev.attr.get("error"):
                self.errs += 1
        return self.errs / self.tools if self.tools else 0.0


class ErrorCount(BaseStream):
    """Absolute number of tool errors, squashed to [0,1]."""
    name = "error_count"

    def __init__(self, cap: int = 5):
        self.cap = cap
        self.errs = 0

    def push(self, ev: Event) -> float:
        if ev.attr.get("error"):
            self.errs += 1
        return min(1.0, self.errs / self.cap)


class RepeatRate(BaseStream):
    """Share of tool calls that repeat an identical (tool, arguments) pair."""
    name = "repeat_rate"

    def __init__(self):
        self.seen: set = set()
        self.tools = 0
        self.reps = 0

    def push(self, ev: Event) -> float:
        if ev.attr.get("kind") == "tool":
            self.tools += 1
            h = ev.attr.get("args_hash")
            if h is not None:
                if h in self.seen:
                    self.reps += 1
                else:
                    self.seen.add(h)
        return self.reps / self.tools if self.tools else 0.0


class EmptyRate(BaseStream):
    """Share of tool calls that returned a well-formed but EMPTY result.

    On tau-bench this is almost entirely a flight search finding nothing.  The
    binary `error` flag misses it (an empty list is not an error), so this is
    the cheap non-bank control for whatever the richer status projection sees.
    """
    name = "empty_rate"

    def __init__(self):
        self.tools = 0
        self.empty = 0

    def push(self, ev: Event) -> float:
        if ev.attr.get("kind") == "tool":
            self.tools += 1
            if ev.attr.get("status") == "empty":
                self.empty += 1
        return self.empty / self.tools if self.tools else 0.0


class PolicyErrorCount(BaseStream):
    """Count of results refusing the action as forbidden, squashed to [0,1].

    "non-pending order cannot be modified" and friends: the environment saying
    the agent just tried something the policy does not allow.  Rare (0.8% of
    results) but, unlike a generic error, unambiguously off-policy.
    """
    name = "policy_count"

    def __init__(self, cap: int = 2):
        self.cap = cap
        self.n = 0

    def push(self, ev: Event) -> float:
        if ev.attr.get("status") in ("policy", "invalid"):
            self.n += 1
        return min(1.0, self.n / self.cap)


class NGramLM(BaseStream):
    """Per-event surprisal of the prefix under a Markov model fit on the bank.

    The natural "learn a process model from the reference set" competitor.
    order 0 = unigram, 1 = bigram.  Scores are mean negative log-probability per
    event, squashed by a fixed scale so they land in [0,1].
    """
    name = "ngram"

    def __init__(self, order: int = 1, alpha: float = 0.5, scale: float = 6.0):
        self.order = order
        self.alpha = alpha
        self.scale = scale
        self.vocab: set = set()
        self.ctx: Dict[tuple, Counter] = defaultdict(Counter)
        self.tot = 0.0
        self.n = 0
        self._prev: Optional[str] = None

    def fit(self, bank: TraceBank) -> "NGramLM":
        for e in bank.entries:
            seq = ["<s>"] * self.order + list(e.labels)
            self.vocab.update(e.labels)
            for i in range(self.order, len(seq)):
                self.ctx[tuple(seq[i - self.order:i])][seq[i]] += 1
        self.vocab.add("<unk>")
        return self

    def _logp(self, ctx: tuple, tok: str) -> float:
        c = self.ctx.get(ctx)
        V = max(1, len(self.vocab))
        if c is None:
            return math.log(1.0 / V)
        num = c.get(tok, 0) + self.alpha
        den = sum(c.values()) + self.alpha * V
        return math.log(num / den)

    def push(self, ev: Event) -> float:
        tok = ev.act
        ctx = () if self.order == 0 else ((self._prev or "<s>"),)
        self.tot += -self._logp(ctx, tok)
        self.n += 1
        self._prev = tok
        return min(1.0, (self.tot / self.n) / self.scale)


class RandomScore(BaseStream):
    """Constant random score per run: the AUC = 0.5 control."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.v = random.Random(seed).random()

    def push(self, ev: Event) -> float:
        return self.v


def make_baselines(bank: TraceBank, seed: int = 0) -> Dict[str, BaseStream]:
    """Fresh baseline scorers for one session."""
    return {
        "step_count": StepCount(),
        "error_rate": ErrorRate(),
        "error_count": ErrorCount(),
        "repeat_rate": RepeatRate(),
        "empty_rate": EmptyRate(),
        "policy_count": PolicyErrorCount(),
        "ngram1": NGramLM(order=1).fit(bank),
        "ngram0": NGramLM(order=0).fit(bank),
        "random": RandomScore(seed=seed),
    }


BASELINE_NAMES = ("step_count", "error_rate", "error_count", "repeat_rate",
                  "empty_rate", "policy_count", "ngram1", "ngram0", "random")
