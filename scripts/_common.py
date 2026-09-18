"""Shared plumbing for the experiments: paths, loading, grouped splits, metrics."""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
TAU = os.path.join(ROOT, "vendor", "tau-bench")
HIST = os.path.join(TAU, "historical_trajectories")

for _d in (DATA, RESULTS, FIGURES):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------- #
# .env  (only the online experiment needs it; loading is harmless otherwise)
# --------------------------------------------------------------------------- #

# Names people actually use -> the names litellm looks for.
KEY_ALIASES = {
    "OPENROUTER_KEY": "OPENROUTER_API_KEY",
    "OPENAI_KEY": "OPENAI_API_KEY",
    "ANTHROPIC_KEY": "ANTHROPIC_API_KEY",
}


def load_dotenv(path: Optional[str] = None) -> List[str]:
    """Read KEY=VALUE lines from .env into the environment.

    Existing environment variables win, so an explicit export always overrides
    the file. Returns the names loaded -- never the values.
    """
    path = path or os.path.join(ROOT, ".env")
    loaded: List[str] = []
    if not os.path.exists(path):
        return loaded
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if not key or not val:
                continue
            for name in {key, KEY_ALIASES.get(key, key)}:
                if not os.environ.get(name):
                    os.environ[name] = val
                    loaded.append(name)
    return loaded

from tracebank.events import Trace  # noqa: E402

# --------------------------------------------------------------------------- #
# Corpora
#
# Two benchmarks.  Rule 5 in HANDOFF_TAU2.md -- never pool across corpora --
# applies with more force here than it did on tau-bench: 16 corpora, 3 domains,
# 6 agent models, 2 benchmarks.  Every table stays per-source.
# --------------------------------------------------------------------------- #

TAU_SOURCES = {
    "gpt-4o/retail": ("gpt-4o-retail.json", "retail", "gpt-4o"),
    "gpt-4o/airline": ("gpt-4o-airline.json", "airline", "gpt-4o"),
    "sonnet-35/retail": ("sonnet-35-new-retail.json", "retail", "sonnet-35"),
    "sonnet-35/airline": ("sonnet-35-new-airline.json", "airline", "sonnet-35"),
}

# tau2: the 12 (model x domain) corpora run with the standard agent and the
# standard user simulator -- `llm_agent` + `user_simulator`, which is the
# faithful analogue of the tau-bench setup.  tau2 also ships `op`
# (llm_agent_gt, an oracle plan), `no-user` (dummy_user) and the
# `telecom-workflow` domain variant; those are tau2's own ablations, not a
# tau-bench analogue, so they are deliberately excluded from the replication.
# `default` and `base` name the same configuration in different result files.
_T2 = "_gpt-4.1-2025-04-14_4trials.json"
TAU2_SOURCES = {
    f"{alias}/{dom}": (f"{mdl}_{dom}_{mode}{_T2}", dom, alias)
    for alias, mdl, mode in (
        ("gpt41",    "gpt-4.1-2025-04-14",           "default"),
        ("gpt41mini", "gpt-4.1-mini-2025-04-14",     "base"),
        ("o4mini",   "o4-mini-2025-04-16",           "default"),
        ("sonnet37", "claude-3-7-sonnet-20250219",   "default"),
    )
    for dom in ("airline", "retail", "telecom")
}

BENCH = {**{k: "tau" for k in TAU_SOURCES},
         **{k: "tau2" for k in TAU2_SOURCES}}

# TB_BENCH=tau | tau2 | all  narrows the default corpus set for every
# experiment at once (each one defaults its --sources to list(SOURCES)).
_WHICH = os.environ.get("TB_BENCH", "all").lower()
if _WHICH == "tau":
    SOURCES = dict(TAU_SOURCES)
elif _WHICH == "tau2":
    SOURCES = dict(TAU2_SOURCES)
elif _WHICH == "all":
    SOURCES = {**TAU_SOURCES, **TAU2_SOURCES}
else:
    raise SystemExit(f"TB_BENCH must be tau|tau2|all, got {_WHICH!r}")


def bench_of(source: str) -> str:
    """Which benchmark a source key belongs to."""
    return BENCH[source]


def load_runs(source: str):
    """Load the raw logged runs for a source, from whichever benchmark it is.

    Returns (runs, domain, model, spec) where `spec` is a tau2 ToolSpec or None
    for tau-bench.  `runs` are the benchmark's own record dicts -- tau-bench
    EnvRunResults or tau2 simulations -- not a normalised shape, because
    normalising would throw away exactly the tau2 fields lambda needs (the
    explicit `error` flag and the tool-call `requestor`).
    """
    fname, domain, model = SOURCES[source]
    if BENCH[source] == "tau":
        with open(os.path.join(HIST, fname)) as f:
            return json.load(f), domain, model, None
    from tracebank.tau2 import load_simulations, spec_for
    sims, _info = load_simulations(fname)
    return sims, domain, model, spec_for(domain)


def trace_builder(source: str):
    """The (fn, domain, model, spec) needed to turn one run into a Trace."""
    from tracebank.abstraction import trace_from_run, trace_from_simulation
    return trace_from_run if BENCH[source] == "tau" else trace_from_simulation


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def traces_path(source: str, granularity: str) -> str:
    return os.path.join(DATA, f"{source.replace('/', '_')}__{granularity}.jsonl")


def save_traces(traces: Sequence[Trace], path: str) -> None:
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(t.to_json()) + "\n")


def load_traces(source: str, granularity: str = "tool") -> List[Trace]:
    p = traces_path(source, granularity)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing -- run experiments/e0_build_dataset.py first")
    with open(p) as f:
        return [Trace.from_json(json.loads(line)) for line in f if line.strip()]


def dump_json(obj, name: str) -> str:
    p = os.path.join(RESULTS, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return p


# --------------------------------------------------------------------------- #
# Splits -- grouped by task_id so a bank trace and an eval trace never share a task
# --------------------------------------------------------------------------- #

@dataclass
class Split:
    bank: List[Trace]
    calib: List[Trace]
    eval: List[Trace]
    seed: int
    # Failing runs from the TRAINING tasks.  Every experiment before E22 is
    # label-free and ignores this; E22's negative arm mines a model from it and
    # is explicitly reported as a supervised ceiling rather than a watchdog.
    # It lives here so the task partition still guarantees no leakage.
    train_fail: List[Trace] = field(default_factory=list)


def grouped_split(traces: Sequence[Trace], k: int, seed: int = 0,
                  train_frac: float = 0.4, leaky: bool = False,
                  per_task: bool = True) -> Split:
    """Draw a k-trace bank from held-out *tasks*.

    Task ids are partitioned first, so no evaluated run comes from a task the
    bank has already seen; this is the honest test of the paper's claim that
    divergence measures the *shape* of a run, not its content.  leaky=True
    disables the partition and gives the optimistic upper bound.
    """
    rng = random.Random(seed)
    task_ids = sorted({t.task_id for t in traces})
    rng.shuffle(task_ids)
    n_train = max(k, int(round(train_frac * len(task_ids))))
    train_ids = set(task_ids[:n_train])
    if leaky:
        train_ids = set(task_ids)

    train = [t for t in traces if t.task_id in train_ids]
    held = traces if leaky else [t for t in traces if t.task_id not in train_ids]

    if per_task:
        by_task: Dict[int, List[Trace]] = {}
        for t in train:
            if t.success:
                by_task.setdefault(t.task_id, []).append(t)
        cand_tasks = sorted(by_task)
        rng.shuffle(cand_tasks)
        bank = [rng.choice(by_task[tid]) for tid in cand_tasks[:k]]
    else:
        successes = [t for t in train if t.success]
        rng.shuffle(successes)
        bank = successes[:k]
    bank_ids = {t.run_id for t in bank}

    calib = [t for t in train if t.run_id not in bank_ids and t.success]
    # never evaluate a trace that is itself in the bank (it would score 0 trivially)
    ev = [t for t in held if t.run_id not in bank_ids]
    train_fail = [t for t in train if not t.success]
    return Split(bank=bank, calib=calib, eval=ev, seed=seed,
                 train_fail=train_fail)


# --------------------------------------------------------------------------- #
# Metrics (no sklearn dependency in the hot path)
# --------------------------------------------------------------------------- #

def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC via the rank (Mann-Whitney) formula; label 1 = the positive class."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[t] = r
        i = j + 1
    npos = sum(l for _, l in pairs)
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    s = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (s - npos * (npos + 1) / 2) / (npos * nneg)


def auc_pr(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    npos = sum(labels)
    if npos == 0:
        return float("nan")
    tp = 0
    ap = 0.0
    for rank, i in enumerate(order, 1):
        if labels[i] == 1:
            tp += 1
            ap += tp / rank
    return ap / npos


def mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    xs = [x for x in xs if x == x]  # drop NaN
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, v ** 0.5


def fmt(m: float, s: float, w: int = 5) -> str:
    return f"{m:.3f}+-{s:.3f}" if m == m else "  n/a "
