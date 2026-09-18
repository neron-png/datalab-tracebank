"""Apply the abstraction function lambda to the logged trajectories.

Covers both benchmarks -- tau-bench's historical_trajectories and tau2's
result files.  Which ones run is controlled by TB_BENCH=tau|tau2|all (default
all) or by --sources.

Run:  python scripts/build_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from _common import (BENCH, SOURCES, dump_json, load_runs, save_traces,
                     traces_path)
from tracebank.abstraction import GRANULARITIES, trace_from_run, trace_from_simulation
from tracebank.events import alphabet

GRANS = GRANULARITIES


def build(source: str, gran: str, runs, domain, model, spec):
    if BENCH[source] == "tau":
        return [trace_from_run(r, domain, model, granularity=gran) for r in runs]
    return [trace_from_simulation(s, domain, model, granularity=gran, spec=spec)
            for s in runs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    a = ap.parse_args()

    summary = {}
    audits = {}
    print(f"{'source':20s} {'bench':6s} {'gran':12s} {'runs':>5s} {'tasks':>5s} "
          f"{'succ':>5s} {'fail':>5s} {'|A|':>4s} {'len med':>8s} {'len succ':>9s} "
          f"{'len fail':>9s}")
    print("-" * 108)
    for source in a.sources:
        runs, domain, model, spec = load_runs(source)
        # HANDOFF Sec 3.4: every tool name must be registered read xor write.
        # Fails loudly on a registry gap; tolerates only names that never once
        # executed cleanly (hallucinated calls, which are agent behaviour).
        if spec is not None:
            from tracebank.tau2 import audit_tool_coverage
            audits[source] = audit_tool_coverage(runs, spec)
        for gran in GRANS:
            traces = [t for t in build(source, gran, runs, domain, model, spec)
                      if len(t) > 0]
            save_traces(traces, traces_path(source, gran))
            succ = [t for t in traces if t.success]
            fail = [t for t in traces if not t.success]
            med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
            row = {
                "bench": BENCH[source], "domain": domain, "model": model,
                "runs": len(traces), "tasks": len({t.task_id for t in traces}),
                "succ": len(succ), "fail": len(fail),
                "fail_rate": round(len(fail) / max(len(traces), 1), 4),
                "alphabet": len(alphabet(traces)),
                "len_median": med([len(t) for t in traces]),
                "len_median_succ": med([len(t) for t in succ]),
                "len_median_fail": med([len(t) for t in fail]),
                "labels": alphabet(traces),
            }
            summary[f"{source}|{gran}"] = row
            print(f"{source:20s} {BENCH[source]:6s} {gran:12s} {row['runs']:5d} "
                  f"{row['tasks']:5d} {row['succ']:5d} {row['fail']:5d} "
                  f"{row['alphabet']:4d} {row['len_median']:8d} "
                  f"{row['len_median_succ']:9d} {row['len_median_fail']:9d}")

    # Sec 5.5 asks for the failure base rate per corpus on the record before
    # anything else is read -- the scores need both classes present.
    print(f"\n{'base rates (gran=tool)':40s}{'runs':>6s}{'fail':>6s}{'fail rate':>11s}")
    print("-" * 63)
    for source in a.sources:
        r = summary[f"{source}|tool"]
        print(f"{source + '  [' + r['bench'] + ']':40s}{r['runs']:6d}"
              f"{r['fail']:6d}{r['fail_rate']:11.3f}")

    if audits:
        print(f"\n{'tau2 tool-name audit':22s}{'names':>7s}{'write':>7s}{'read':>6s}"
              f"  hallucinated (never executed cleanly)")
        print("-" * 100)
        for source, au in audits.items():
            h = au["hallucinated"]
            tot = sum(h.values())
            names = ", ".join(sorted(h)[:4]) + (" ..." if len(h) > 4 else "")
            print(f"{source:22s}{au['n_tool_names']:7d}{au['n_write']:7d}"
                  f"{au['n_read']:6d}  {tot:4d} calls over {len(h):2d} names"
                  + (f": {names}" if h else ""))
        summary["_tau2_tool_audit"] = audits

    p = dump_json(summary, "e0_corpus.json")
    print(f"\ntraces written to data/, summary -> {p}")


if __name__ == "__main__":
    main()
