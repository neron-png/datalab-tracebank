"""Figure: mean F1 and earliness per detector, with across-environment error bars.

Reads results/e31_quantile_threshold_paper.json and writes
figures/detector_f1_earliness.{pdf,png}.

The environment is the unit of replication. Each detector gets one F1 per
environment, averaged over that environment's 3 bank sizes x 3 quantiles; the
bar is the mean of those 16 values and the whisker their standard error. The
9 settings inside an environment rescore the same traces, so they are not
independent and dividing by sqrt(144) would understate the error.

Cells where a detector never fires have undefined F1 and count as zero.
Earliness is defined only over true-positive flags, so undefined cells are
skipped there instead.

Run:  python scripts/fig_detector_bars.py
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "e31_quantile_threshold_paper.json")
OUT = os.path.join(ROOT, "figures", "detector_f1_earliness")

WARMUP_KEY = "w1"                      # LOO peaks collected from w = 3 events

BANK, STRUCT, DECL, BASE = "bank", "struct", "decl", "base"
COLOR = {BANK: "#2166ac", STRUCT: "#b2182b", DECL: "#4dac26", BASE: "#878787"}
FAMILY_LABEL = {BANK: "Bank divergence", STRUCT: "Structural divergence",
                DECL: "Declarative conformance", BASE: "Step-count baseline"}

# detector key -> (display label, family). dl/q25, edge novelty and the two
# standardised |z| detectors are excluded from the paper.
DETECTORS = [
    ("dl/max@3",      "DL max-$\\kappa$",            BANK),
    ("dltr/max@3",    "DL trunc. max-$\\kappa$",     BANK),
    ("dl/min",        "DL min",                      BANK),
    ("dl/mean@3",     "DL mean-$\\kappa$",           BANK),
    ("jac/max@3",     "Jaccard max-$\\kappa$",       BANK),
    ("jac/min",       "Jaccard min",                 BANK),
    ("jac_pre/max@3", "Prefix-Jaccard max-$\\kappa$", BANK),
    ("cont/min",      "Containment min",             BANK),
    ("return_rate",   "Return rate",                 STRUCT),
    ("df_legality",   "DF legality",                 STRUCT),
    ("graph_progress", "Graph progress",             STRUCT),
    ("dcl_viol",      "Constraint violation",        DECL),
    ("dcl_div",       "Constraint divergence",       DECL),
]
BASELINE = ("step_count", "Step count (baseline)", BASE)


def per_environment(results, det, metric, zero_if_silent):
    """One value per environment, averaged over its 3 x 3 (k, alpha) settings."""
    out = []
    for env in results:
        vals = []
        for k in results[env]:
            if det not in results[env][k]:
                continue
            for q in results[env][k][det]:
                v = results[env][k][det][q][WARMUP_KEY][metric][0]
                if v == v:
                    vals.append(v)
                elif zero_if_silent:
                    vals.append(0.0)
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def summarise(results, det):
    out = []
    for metric, zero in (("f1", True), ("earliness_frac", False)):
        vals = per_environment(results, det, metric, zero)
        if not vals:
            out.append((float("nan"), float("nan")))
            continue
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            out.append((m, 0.0))
            continue
        var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        out.append((m, math.sqrt(var / len(vals))))      # standard error
    return tuple(out)


def whiskers(means, sds):
    """Symmetric sd, lower arm clipped at zero since both metrics are non-negative."""
    lo = [min(m, s) if m == m else 0.0 for m, s in zip(means, sds)]
    hi = [0.0 if s != s else s for s in sds]
    return [lo, hi]


def main():
    results = json.load(open(SRC))["results"]
    stats = {d: summarise(results, d) for d, _, _ in DETECTORS + [BASELINE]}

    ranked = sorted(DETECTORS, key=lambda d: stats[d[0]][0][0], reverse=True)
    rows = ranked + [BASELINE]                    # baseline sits on top
    labels = [lab for _, lab, _ in rows]
    colors = [COLOR[fam] for _, _, fam in rows]
    ys = list(range(len(rows)))                   # best at the bottom

    f1_m = [stats[d][0][0] for d, _, _ in rows]
    f1_s = [stats[d][0][1] for d, _, _ in rows]
    e_m = [stats[d][1][0] for d, _, _ in rows]
    e_s = [stats[d][1][1] for d, _, _ in rows]
    base_f1, base_e = stats[BASELINE[0]][0][0], stats[BASELINE[0]][1][0]

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    bar = dict(height=0.62, zorder=2)
    err = dict(ecolor="#333333", elinewidth=0.9, capsize=2.2, zorder=3)

    ax1.barh(ys, f1_m, color=colors, xerr=whiskers(f1_m, f1_s),
             error_kw=err, **bar)
    ax1.axvline(base_f1, color="#a5a5a5", ls="--", lw=1.0, zorder=1)
    ax1.set_title("(a) $F_1$ score")
    ax1.set_xlabel("Mean $F_1$")
    ax1.set_xlim(0, max(m + s for m, s in zip(f1_m, f1_s)) * 1.08)

    ax2.barh(ys, [0.0 if m != m else m for m in e_m],
             color=colors, xerr=whiskers(e_m, e_s), error_kw=err, **bar)
    ax2.axvline(base_e, color="#a5a5a5", ls="--", lw=1.0, zorder=1)
    ax2.set_title("(b) Earliness")
    ax2.set_xlabel("Mean earliness (fraction of run)")
    ax2.set_xlim(0, 1.0)

    ax1.set_yticks(ys)
    ax1.set_yticklabels(labels)
    ax1.set_ylim(-0.7, len(rows) - 0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[f])
               for f in (BANK, STRUCT, DECL, BASE)]
    fig.legend(handles, [FAMILY_LABEL[f] for f in (BANK, STRUCT, DECL, BASE)],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}.pdf\n")

    for det, lab, _ in rows:
        (fm, fs), (em, es) = stats[det]
        print(f"{lab:26s} F1 {fm:.3f} +- {fs:.3f}   earl {em:.3f} +- {es:.3f}")


if __name__ == "__main__":
    main()
