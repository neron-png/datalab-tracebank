"""E31: quantile-thresholded LOO calibration — the paper's definitive sweep.

Run-level split (no task partition), all detector families from the paper:
  - Bank divergence: edit distance, Jaccard, containment (various aggregations)
  - Structural: return rate, edge novelty, DF legality, graph progress,
                standardised |z| of return rate and edge novelty
  - Declarative: constraint violation, constraint divergence (Jaccard in
                 constraint-signature space)
  - Baseline: step count

Excludes detectors not discussed in the paper (ngram, cosine variants).

Run:  python scripts/run_experiment.py [--seeds 10] [--sources ...]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from _common import (SOURCES, auc_roc, dump_json, fmt, grouped_split,
                     load_traces, mean_std)
from _detectors import decide, stream
from tracebank.bank import ScoreConfig, SessionScorer, TraceBank
from tracebank.baselines import StepCount
from tracebank.declare import (COUNTING_TEMPLATES, DEFAULT_EXCLUDED,
                               DeclareModel, DeclareScorer, mine_local)
from tracebank.declare_bank import DeclareBankScorer, SignatureBank
from tracebank.divmetrics import DFLegality
from tracebank.events import Trace
from tracebank.graphmetrics import (CycleAbsZ, EdgeAbsZ, EdgeNovelty,
                                    GraphProgress, ReturnRate)

QUANTILES = (0.90, 0.95, 0.99)
KS = (16, 32, 64)
WARMUP = 3
PATIENCE = 2

TB_CONFIGS = [
    ScoreConfig("dl", "max_knn", kappa=3, name="dl/max@3"),
    ScoreConfig("dl_trunc", "max_knn", kappa=3, name="dltr/max@3"),
    ScoreConfig("dl", "min", name="dl/min"),
    ScoreConfig("dl", "mean_knn", kappa=3, name="dl/mean@3"),
    ScoreConfig("dl", "quantile", q=0.25, name="dl/q25"),
    ScoreConfig("jaccard", "max_knn", kappa=3, name="jac/max@3"),
    ScoreConfig("jaccard", "min", name="jac/min"),
    ScoreConfig("jaccard_prefix", "max_knn", kappa=3, name="jac_pre/max@3"),
    ScoreConfig("containment", "min", name="cont/min"),
]

STRUCTURAL_NAMES = [
    "return_rate", "edge_novelty", "df_legality", "graph_progress",
    "cycle_absz", "edge_absz",
]

DECLARE_NAMES = [
    "dcl_viol",     # constraint violation (Sec 4.4)
    "dcl_div",      # constraint divergence — Jaccard in signature space (Sec 4.4)
]

DET_ORDER = (
    [c.label() for c in TB_CONFIGS]
    + STRUCTURAL_NAMES
    + DECLARE_NAMES
    + ["step_count"]
)

MakeDet = Callable[[TraceBank, Sequence[Trace]], Callable[[], object]]


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("inf")
    values = sorted(values)
    idx = min(len(values) - 1, int(math.ceil(q * (len(values) - 1))))
    return values[idx]


def loo_peaks(bank_traces: Sequence[Trace], make_det: MakeDet,
              warmup: int) -> Tuple[List[float], List[float]]:
    peaks_w1: List[float] = []
    peaks_w3: List[float] = []
    w3 = 3 * warmup
    for i in range(len(bank_traces)):
        loo = list(bank_traces[:i]) + list(bank_traces[i + 1:])
        loo_bank = TraceBank.from_traces(loo)
        det = make_det(loo_bank, loo)
        scores = stream(det, bank_traces[i])
        if len(scores) >= warmup:
            peaks_w1.append(max(scores[warmup - 1:]))
        if len(scores) >= w3:
            peaks_w3.append(max(scores[w3 - 1:]))
    return peaks_w1, peaks_w3


def eval_metrics(curves: List[List[float]], y: List[int],
                 threshold: float, warmup: int,
                 patience: int) -> Dict[str, float]:
    dec = [decide(c, len(c), threshold, warmup, patience) for c in curves]
    tp = sum(1 for d, l in zip(dec, y) if d.fired and l == 1)
    fp = sum(1 for d, l in zip(dec, y) if d.fired and l == 0)
    npos = sum(y)
    nneg = len(y) - npos
    early_frac = [d.frac for d, l in zip(dec, y) if d.fired and l == 1]
    early_step = [d.step for d, l in zip(dec, y) if d.fired and l == 1]
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / npos if npos else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fp / nneg if nneg else float("nan"),
        "flag_rate": (tp + fp) / len(y),
        "earliness_frac": (sum(early_frac) / len(early_frac)
                           if early_frac else float("nan")),
        "earliness_step": (sum(early_step) / len(early_step)
                           if early_step else float("nan")),
    }


def _mine_declare(bank_traces: Sequence[Trace]) -> Optional[DeclareModel]:
    """Mine a DECLARE model from the bank alone (no failure labels).

    Constraints that hold on every bank trace (support=1.0) form the model.
    Counting templates are excluded because they proxy run length.
    Returns None when mining yields no usable constraints.
    """
    if not bank_traces:
        return None
    exclude = set(DEFAULT_EXCLUDED) | set(COUNTING_TEMPLATES) | {"init"}
    cons = mine_local(bank_traces, support_threshold=1.0,
                      cumulative_existential=True)
    cons = [c for c in cons if c.template not in exclude]
    if not cons:
        return None
    return DeclareModel(cons)


Key = Tuple[str, int, str, float, str]


def run_seed(sources: Sequence[str], trace_cache: Dict[str, List[Trace]],
             seed: int, warmup: int, patience: int,
             acc: Dict[Key, Dict[str, List[float]]],
             skipped: List[Dict]) -> None:
    for source in sources:
        traces = trace_cache[source]
        for k in KS:
            sp = grouped_split(traces, k=k, seed=seed,
                               per_task=False, leaky=True)
            if len(sp.bank) < k:
                if seed == 0:
                    skipped.append({"source": source, "k": k,
                                    "reason": f"only {len(sp.bank)} bank traces"})
                continue

            ev = [t for t in sp.eval if len(t) >= warmup]
            y = [0 if t.success else 1 for t in ev]
            if not (0 < sum(y) < len(y)):
                if seed == 0:
                    skipped.append({"source": source, "k": k,
                                    "reason": "no class variation in eval"})
                continue

            full_bank = TraceBank.from_traces(sp.bank)

            dcl_model = _mine_declare(sp.bank)

            raw_bank = sp.bank

            det_factories: Dict[str, MakeDet] = {}

            for c in TB_CONFIGS:
                det_factories[c.label()] = (
                    lambda b, rt, _c=c: lambda: SessionScorer(b, _c))

            det_factories["df_legality"] = lambda b, rt: lambda: DFLegality(b)
            det_factories["graph_progress"] = lambda b, rt: lambda: GraphProgress(b)

            det_factories["return_rate"] = lambda b, rt: lambda: ReturnRate()
            det_factories["edge_novelty"] = lambda b, rt: lambda: EdgeNovelty()
            det_factories["cycle_absz"] = lambda b, rt: lambda: CycleAbsZ(rt)
            det_factories["edge_absz"] = lambda b, rt: lambda: EdgeAbsZ(rt)

            det_factories["step_count"] = lambda b, rt: lambda: StepCount()

            if dcl_model is not None:
                _dm = dcl_model
                det_factories["dcl_viol"] = (
                    lambda b, rt, m=_dm: lambda: DeclareScorer(
                        "dcl_pos_viol", pos=m, name="dcl_viol"))
                det_factories["dcl_div"] = (
                    lambda b, rt, m=_dm: lambda: DeclareBankScorer(
                        SignatureBank(m, b.entries, "state", mode="prefix"),
                        kappa=3, aggregator="max_knn", name="dcl_div"))

            for det_name in DET_ORDER:
                if det_name not in det_factories:
                    continue
                make = det_factories[det_name]

                full_det = make(full_bank, raw_bank)
                curves = [stream(full_det, t) for t in ev]

                pw1, pw3 = loo_peaks(sp.bank, make, warmup)

                for q in QUANTILES:
                    thr_w1 = _quantile(pw1, q)
                    thr_w3 = _quantile(pw3, q)

                    m1 = eval_metrics(curves, y, thr_w1, warmup, patience)
                    m3 = eval_metrics(curves, y, thr_w3, 3 * warmup, patience)

                    key_w1: Key = (source, k, det_name, q, "w1")
                    key_w3: Key = (source, k, det_name, q, "w3")
                    for metric, val in m1.items():
                        acc[key_w1][metric].append(val)
                    acc[key_w1]["threshold"].append(thr_w1)
                    for metric, val in m3.items():
                        acc[key_w3][metric].append(val)
                    acc[key_w3]["threshold"].append(thr_w3)


def print_preview(acc: Dict[Key, Dict[str, List[float]]],
                  sources: Sequence[str]) -> None:
    print("\n" + "=" * 90)
    print("SEED 0 PREVIEW")
    print("=" * 90)

    for source in sources:
        for k in KS:
            rows = [(det, q, wt)
                    for det in DET_ORDER
                    for q in QUANTILES
                    for wt in ("w1", "w3")
                    if (source, k, det, q, wt) in acc]
            if not rows:
                continue

            print(f"\n--- {source}  k={k} ---")
            print(f"  {'detector':20s} {'q':>5s} {'w':>3s} {'prec':>7s} "
                  f"{'recall':>7s} {'f1':>7s} {'fpr':>7s} {'flag%':>7s} "
                  f"{'early':>7s} {'thr':>7s}")

            for det in DET_ORDER:
                for q in QUANTILES:
                    for wt in ("w1", "w3"):
                        key = (source, k, det, q, wt)
                        if key not in acc:
                            continue
                        d = acc[key]
                        def g(m):
                            v = d[m]
                            return f"{v[0]:.3f}" if v and v[0] == v[0] else "  n/a"
                        print(f"  {det:20s} {q:5.2f} {wt:>3s} {g('precision'):>7s} "
                              f"{g('recall'):>7s} {g('f1'):>7s} {g('fpr'):>7s} "
                              f"{g('flag_rate'):>7s} {g('earliness_frac'):>7s} "
                              f"{g('threshold'):>7s}")


def build_output(acc: Dict[Key, Dict[str, List[float]]],
                 sources: Sequence[str], raw: bool = False) -> Dict:
    out: Dict = {}
    for source in sources:
        src_d: Dict = {}
        for k in KS:
            k_d: Dict = {}
            for det in DET_ORDER:
                det_d: Dict = {}
                for q in QUANTILES:
                    q_d: Dict = {}
                    for wt in ("w1", "w3"):
                        key = (source, k, det, q, wt)
                        if key not in acc:
                            continue
                        cell = {m: list(mean_std(v))
                                for m, v in acc[key].items()}
                        if raw:
                            # per-seed values in seed order, for aggregating
                            # seed-major across environments and settings
                            cell["raw"] = {m: list(v)
                                           for m, v in acc[key].items()}
                        q_d[wt] = cell
                    if q_d:
                        det_d[str(q)] = q_d
                if det_d:
                    k_d[det] = det_d
            if k_d:
                src_d[str(k)] = k_d
        if src_d:
            out[source] = src_d
    return out


def print_final(acc: Dict[Key, Dict[str, List[float]]],
                sources: Sequence[str]) -> None:
    print("\n" + "=" * 100)
    print("FINAL RESULTS (mean +- std over all seeds)")
    print("=" * 100)

    for source in sources:
        for k in KS:
            rows = [(det, q, wt)
                    for det in DET_ORDER
                    for q in QUANTILES
                    for wt in ("w1",)
                    if (source, k, det, q, wt) in acc]
            if not rows:
                continue

            print(f"\n--- {source}  k={k} ---")
            print(f"  {'detector':20s} {'q':>5s} {'prec':>14s} "
                  f"{'recall':>14s} {'f1':>14s} {'fpr':>14s} {'early':>14s}")

            for det in DET_ORDER:
                for q in QUANTILES:
                    key = (source, k, det, q, "w1")
                    if key not in acc:
                        continue
                    d = acc[key]
                    print(f"  {det:20s} {q:5.2f} "
                          f"{fmt(*mean_std(d['precision'])):>14s} "
                          f"{fmt(*mean_std(d['recall'])):>14s} "
                          f"{fmt(*mean_std(d['f1'])):>14s} "
                          f"{fmt(*mean_std(d['fpr'])):>14s} "
                          f"{fmt(*mean_std(d['earliness_frac'])):>14s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--raw", action="store_true",
                    help="also dump per-seed values for each cell")
    ap.add_argument("--out", default="e31_quantile_threshold_paper.json")
    a = ap.parse_args()

    print(f"E31 quantile threshold (paper sweep): {len(a.sources)} sources, "
          f"k in {KS}, q in {QUANTILES}, {a.seeds} seeds")
    print(f"Detectors: {len(DET_ORDER)} ({len(TB_CONFIGS)} bank + "
          f"{len(STRUCTURAL_NAMES)} structural + "
          f"{len(DECLARE_NAMES)} declare + step_count)")
    print(f"Split: run-level (leaky=True, per_task=False)")
    t0 = time.time()

    trace_cache = {}
    for source in a.sources:
        trace_cache[source] = load_traces(source, "tool")

    acc: Dict[Key, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    skipped: List[Dict] = []

    seed_order = [0] + list(range(1, a.seeds))
    for seed in seed_order:
        print(f"\n--- seed {seed} ---", flush=True)
        run_seed(a.sources, trace_cache, seed, a.warmup, a.patience,
                 acc, skipped)
        if seed == 0:
            print_preview(acc, a.sources)
            sys.stdout.flush()

    if skipped:
        print(f"\nSkipped {len(skipped)} combinations:")
        for s in skipped:
            print(f"  {s['source']} k={s['k']}: {s['reason']}")

    print_final(acc, a.sources)

    elapsed = time.time() - t0
    base_rates = {src: sum(1 for t in ts if not t.success) / len(ts)
                  for src, ts in trace_cache.items()}

    p = dump_json({
        "config": {
            "seeds": a.seeds, "warmup": a.warmup, "patience": a.patience,
            "ks": list(KS), "quantiles": list(QUANTILES),
            "detectors": DET_ORDER, "sources": a.sources,
            "bank_mode": "per_run", "split": "leaky",
        },
        "base_rates": base_rates,
        "skipped": skipped,
        "elapsed_s": round(elapsed, 1),
        "results": build_output(acc, a.sources, raw=a.raw),
    }, a.out)
    print(f"\nsaved -> {p}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
