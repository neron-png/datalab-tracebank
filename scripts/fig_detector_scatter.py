"""Figures: F1 versus earliness scatter for the paper's 13 detectors + baseline.

Reads results/e31_quantile_threshold_paper.json and writes four figures into
figures/:

  detector_scatter_env.pdf      one panel per environment (16)
  detector_scatter_domain.pdf   one panel per domain (airline, retail, telecom)
  detector_scatter_overall.pdf  a single panel, the scatter form of Figure 2
  detector_scatter_best.pdf     the same, each detector at its own best (k, alpha)

Aggregation follows fig_detector_bars.py. Each detector gets one F1 and one
earliness per environment, averaged over that environment's 3 bank sizes x 3
quantiles at warmup w = 3. Cells where a detector never fires have undefined F1
and count as zero; earliness is defined only over true positives, so undefined
cells are skipped there. Domain and overall panels average those per-environment
values and draw the standard error across environments as whiskers.

Colour marks the detector family, marker shape separates detectors inside a
family, so identity never rests on colour alone. Up and to the left is better.

Run:  python scripts/fig_detector_scatter.py
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "e31_quantile_threshold_paper.json")
OUTDIR = os.path.join(ROOT, "figures")

WARMUP_KEY = "w1"                      # LOO peaks collected from w = 3 events

BANK, STRUCT, DECL, BASE = "bank", "struct", "decl", "base"
COLOR = {BANK: "#2166ac", STRUCT: "#b2182b", DECL: "#4dac26", BASE: "#878787"}
FAMILY_LABEL = {BANK: "Bank divergence", STRUCT: "Structural divergence",
                DECL: "Declarative conformance", BASE: "Step-count baseline"}

# detector key -> (display label, family, marker). dl/q25, edge novelty and the
# two standardised |z| detectors are excluded from the paper.
DETECTORS = [
    ("dl/max@3",      "DL max-$\\kappa$",             BANK,   "o"),
    ("dltr/max@3",    "DL trunc. max-$\\kappa$",      BANK,   "s"),
    ("dl/min",        "DL min",                       BANK,   "^"),
    ("dl/mean@3",     "DL mean-$\\kappa$",            BANK,   "v"),
    ("jac/max@3",     "Jaccard max-$\\kappa$",        BANK,   "D"),
    ("jac/min",       "Jaccard min",                  BANK,   "P"),
    ("jac_pre/max@3", "Prefix-Jaccard max-$\\kappa$", BANK,   "X"),
    ("cont/min",      "Containment min",              BANK,   "*"),
    ("return_rate",   "Return rate",                  STRUCT, "o"),
    ("df_legality",   "DF legality",                  STRUCT, "s"),
    ("graph_progress", "Graph progress",              STRUCT, "^"),
    ("dcl_viol",      "Constraint violation",         DECL,   "o"),
    ("dcl_div",       "Constraint divergence",        DECL,   "s"),
    ("step_count",    "Step count (baseline)",        BASE,   "h"),
]

MODEL_LABEL = {"gpt-4o": "GPT-4o", "sonnet-35": "Sonnet 3.5",
               "gpt41": "GPT-4.1", "gpt41mini": "GPT-4.1-mini",
               "o4mini": "o4-mini", "sonnet37": "Sonnet 3.7"}
TAU_MODELS = {"gpt-4o", "sonnet-35"}
DOMAINS = ("airline", "retail", "telecom")

# environments in the reading order of Table 3: tau first, then tau2, by domain
ENV_ORDER = [
    "gpt-4o/airline", "sonnet-35/airline", "gpt-4o/retail", "sonnet-35/retail",
    "gpt41/airline", "gpt41mini/airline", "o4mini/airline", "sonnet37/airline",
    "gpt41/retail", "gpt41mini/retail", "o4mini/retail", "sonnet37/retail",
    "gpt41/telecom", "gpt41mini/telecom", "o4mini/telecom", "sonnet37/telecom",
]


def env_label(env):
    model, domain = env.split("/")
    bench = "$\\tau$" if model in TAU_MODELS else "$\\tau^2$"
    return f"{bench}-{domain} · {MODEL_LABEL[model]}"


def cell_mean(results, env, det, metric, zero_if_silent):
    """One value for a detector in one environment, over its 3 x 3 settings."""
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
    return sum(vals) / len(vals) if vals else float("nan")


def per_environment(results, det):
    """{env: (f1, earliness)} for every environment the detector ran on."""
    out = {}
    for env in results:
        f1 = cell_mean(results, env, det, "f1", True)
        earl = cell_mean(results, env, det, "earliness_frac", False)
        if f1 == f1:
            out[env] = (f1, earl)
    return out


def mean_sem(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var / len(vals))


def style():
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.grid": True, "grid.color": "#e6e6e6",
                         "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})


def draw(ax, points, size=46):
    """points: {det_key: (f1, earliness, f1_err, earl_err)}; errors may be None."""
    for key, label, fam, marker in DETECTORS:
        if key not in points:
            continue
        f1, earl, f1e, earle = points[key]
        if earl != earl:
            continue
        if f1e is not None:
            ax.errorbar(earl, f1, xerr=earle, yerr=f1e, fmt="none",
                        ecolor=COLOR[fam], elinewidth=0.8, capsize=1.8,
                        alpha=0.55, zorder=2)
        ax.scatter(earl, f1, s=size * (1.7 if marker == "*" else 1.0),
                   marker=marker, color=COLOR[fam], edgecolors="white",
                   linewidths=0.6, zorder=3)


def legend_handles(labels=None):
    return [Line2D([0], [0], marker=m, color="none", markerfacecolor=COLOR[f],
                   markeredgecolor="white", markeredgewidth=0.5,
                   markersize=8 if m != "*" else 10,
                   label=labels[key] if labels else lab)
            for key, lab, f, m in DETECTORS]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(OUTDIR, name)}.pdf")


def fig_per_env(stats):
    fig, axes = plt.subplots(4, 4, figsize=(11.5, 10.5), sharex=True, sharey=True)
    for ax, env in zip(axes.flat, ENV_ORDER):
        pts = {k: (stats[k][env][0], stats[k][env][1], None, None)
               for k, _, _, _ in DETECTORS if env in stats[k]}
        draw(ax, pts, size=40)
        ax.set_title(env_label(env), fontsize=8.5, pad=4)
        ax.set_xlim(-0.02, 1.03)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Earliness (fraction of run)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("$F_1$", fontsize=8.5)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.045), fontsize=8.5,
               handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout()
    save(fig, "detector_scatter_env")


def fig_per_domain(stats):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        pts = {}
        for key, _, _, _ in DETECTORS:
            envs = [e for e in stats[key] if e.endswith("/" + dom)]
            if not envs:
                continue
            f1, f1e = mean_sem([stats[key][e][0] for e in envs])
            earl, earle = mean_sem([stats[key][e][1] for e in envs])
            pts[key] = (f1, earl, f1e, earle)
        draw(ax, pts)
        n = len({e for e in stats["dl/min"] if e.endswith("/" + dom)})
        ax.set_title(f"{dom} ({n} environments)", fontsize=9.5)
        ax.set_xlabel("Mean earliness (fraction of run)")
        ax.set_xlim(-0.02, 1.03)
    axes[0].set_ylabel("Mean $F_1$")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.22), fontsize=8.5,
               handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout()
    save(fig, "detector_scatter_domain")


def fig_overall(stats):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    pts, rows = {}, []
    for key, label, _, _ in DETECTORS:
        envs = sorted(stats[key])
        f1, f1e = mean_sem([stats[key][e][0] for e in envs])
        earl, earle = mean_sem([stats[key][e][1] for e in envs])
        pts[key] = (f1, earl, f1e, earle)
        rows.append((label, f1, f1e, earl, earle))
    draw(ax, pts, size=64)
    ax.set_xlabel("Mean earliness (fraction of run)")
    ax.set_ylabel("Mean $F_1$")
    ax.set_xlim(-0.02, 1.03)
    ax.annotate("better", xy=(0.06, 0.95), xycoords="axes fraction",
                xytext=(0.22, 0.95), textcoords="axes fraction",
                fontsize=8, color="#777777", va="center",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.9))
    ax.legend(handles=legend_handles(), loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8.5, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "detector_scatter_overall")

    print()
    for label, f1, f1e, earl, earle in rows:
        print(f"{label:30s} F1 {f1:.3f} +- {f1e:.3f}   earl {earl:.3f} +- {earle:.3f}")


def setting_values(results, det, k, q, metric, zero_if_silent):
    """One value per environment for a detector held at a fixed (k, alpha)."""
    out = []
    for env in results:
        cell = results[env].get(k, {}).get(det, {}).get(q)
        if cell is None:
            continue
        v = cell[WARMUP_KEY][metric][0]
        if v == v:
            out.append(v)
        elif zero_if_silent:
            out.append(0.0)
    return out


def best_setting(results, det):
    """The (k, alpha) with the highest F1 averaged over the 16 environments."""
    best = None
    for k in sorted({k for env in results for k in results[env]}, key=int):
        for q in sorted({q for env in results for q in results[env].get(k, {}).get(det, {})},
                        key=float):
            f1, f1e = mean_sem(setting_values(results, det, k, q, "f1", True))
            if f1 != f1:
                continue
            if best is None or f1 > best[2]:
                earl, earle = mean_sem(
                    setting_values(results, det, k, q, "earliness_frac", False))
                best = (k, q, f1, f1e, earl, earle)
    return best


def fig_best_setting(results):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    pts, labels, rows = {}, {}, []
    for key, label, _, _ in DETECTORS:
        k, q, f1, f1e, earl, earle = best_setting(results, key)
        pts[key] = (f1, earl, f1e, earle)
        labels[key] = f"{label} ({k}@{float(q):g})"
        rows.append((labels[key], f1, f1e, earl, earle))
    draw(ax, pts, size=64)
    ax.set_xlabel("Mean earliness (fraction of run)")
    ax.set_ylabel("Mean $F_1$ at best $(k, \\alpha)$")
    ax.set_xlim(-0.02, 1.03)
    ax.annotate("better", xy=(0.06, 0.95), xycoords="axes fraction",
                xytext=(0.22, 0.95), textcoords="axes fraction",
                fontsize=8, color="#777777", va="center",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.9))
    ax.legend(handles=legend_handles(labels), loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8.5, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "detector_scatter_best")

    print()
    for label, f1, f1e, earl, earle in rows:
        print(f"{label:38s} F1 {f1:.3f} +- {f1e:.3f}   earl {earl:.3f} +- {earle:.3f}")


def main():
    results = json.load(open(SRC))["results"]
    stats = {k: per_environment(results, k) for k, _, _, _ in DETECTORS}
    missing = [e for e in results if e not in ENV_ORDER]
    if missing or len(ENV_ORDER) != len(results):
        raise SystemExit(f"environment list out of sync: {missing}")
    style()
    fig_per_env(stats)
    fig_per_domain(stats)
    fig_overall(stats)
    fig_best_setting(results)


if __name__ == "__main__":
    main()
