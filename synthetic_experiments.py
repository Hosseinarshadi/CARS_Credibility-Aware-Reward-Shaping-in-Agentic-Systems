"""Synthetic experiments: E1 (P1 credibility boundary), E2 (P2 collective verification),
E3s (P3 policy value under bandit learners) and E4 (ablation of credibility dimensions).

Usage:
    python -m experiments.synthetic_experiments --exp all [--quick]
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pandas as pd

from cars.synthetic.policies import ABLATIONS, MAIN_POLICIES, PolicyConfig
from cars.synthetic.sim import SimConfig, predicted_lifetime, run

from .common import (POLICY_COLORS, POLICY_LABELS, diff_ci, fmt_ci, load_jsonl, mean_ci, plt, run_many,
                     save_jsonl, savefig, settings, write_table)

ATTACKERS_ALL = ["UCB1", "Thompson", "EXP3", "eps-decay", "eps-const", "Suspicious-UCB", "Skeptical-UCB", "BARBAR"]


def _run(cfg: SimConfig) -> dict:
    return run(cfg)


def _group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


# ============================================================== E1 / P1
def e1(quick: bool):
    S = settings(quick)
    R = S["runs"]
    fs = [0.3, 0.5, 0.7, 0.9]
    vs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    # (a) lifetime: decoy-only deception, no rotation, so the first exposure is the measurement
    cfgs = []
    for f in fs:
        pol = PolicyConfig(f"decoy_f{f}", decoy_rate=f, sem=True)
        for v in vs:
            for att in ["UCB1", "Thompson"]:
                for s in range(R * 2):
                    c = SimConfig(policy_override=pol, attacker=att, budget=S["budget_small"], v_att=v, seed=s)
                    cfgs.append(c)
    rows = run_many(_run, cfgs, "E1a lifetime")
    for r, c in zip(rows, cfgs):
        r["f"] = c.policy_override.decoy_rate
    save_jsonl(rows, "e1a_lifetime")

    recs = []
    for (f, v, att), rs in sorted(_group(rows, ["f", "v_att", "attacker"]).items()):
        obs = [r["decoy_pulls_before_decoy_exposure"] for r in rs]
        censored = sum(o is None for o in obs)
        m, lo, hi = mean_ci([o for o in obs if o is not None])
        recs.append({"f": f, "v": v, "attacker": att, "pred": predicted_lifetime(f, v, 1.0),
                     "obs": m, "lo": lo, "hi": hi, "censored": censored, "n": len(obs)})
    df = pd.DataFrame(recs)
    tab = df[df.attacker == "UCB1"].copy()
    tab["Predicted 1/h"] = tab.pred.map(lambda x: f"{x:.1f}")
    tab["Observed [95% CI]"] = [fmt_ci(a, b, c, 1) for a, b, c in zip(tab.obs, tab.lo, tab.hi)]
    tab["Obs/Pred"] = (tab.obs / tab.pred).map(lambda x: f"{x:.2f}")
    write_table(tab[["f", "v", "Predicted 1/h", "Observed [95% CI]", "Obs/Pred", "censored"]]
                .rename(columns={"f": "Fabricated rate f", "v": "Verification prob. v", "censored": "Censored runs"}),
                "table_e1_lifetime",
                "P1: decoy interactions until exposure, predicted (1/h, h = f v v_b) vs observed (UCB1 attacker).",
                "tab:e1-lifetime")

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for att, mk in [("UCB1", "o"), ("Thompson", "s")]:
        d = df[df.attacker == att]
        ax.errorbar(d.pred, d.obs, yerr=[d.obs - d.lo, d.hi - d.obs], fmt=mk, ms=3.5, lw=0.8,
                    label=att, alpha=0.85)
    lim = [1, max(df.pred.max(), df.hi.max()) * 1.3]
    ax.plot(lim, lim, "k--", lw=0.8, label="Observed = predicted")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Predicted lifetime 1/h (decoy interactions)")
    ax.set_ylabel("Observed lifetime")
    ax.set_title("P1: lifetime follows 1/(f·v·v_b)")
    ax.legend(frameon=False)
    savefig(fig, "fig_e1a_lifetime")

    # (b) boundary: value of deception vs verification probability, with and without rotation
    vs_b = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    atts = ATTACKERS_ALL if not quick else ["UCB1", "Suspicious-UCB"]
    cfgs = [SimConfig(policy=p, attacker=a, budget=S["budget"], v_att=v, seed=s)
            for a in atts for v in vs_b for p in ["block", "cars_noverif", "cars"] for s in range(R)]
    rows = run_many(_run, cfgs, "E1b boundary")
    for r in rows:
        r.pop("curve", None)
    save_jsonl(rows, "e1b_boundary")
    g = _group(rows, ["attacker", "v_att", "policy"])
    recs = []
    for a in atts:
        for v in vs_b:
            cb = np.mean([r["C"] for r in g[(a, v, "block")]])
            for p in ["cars_noverif", "cars"]:
                cp = np.mean([r["C"] for r in g[(a, v, p)]])
                recs.append({"attacker": a, "v": v, "policy": p, "C_block": cb, "C": cp,
                             "reduction": cb / max(cp, 1e-9)})
    df = pd.DataFrame(recs)
    plot_e1b(df, atts)
    bnd = []
    for a in atts:
        row = {"Attacker": a}
        for p in ["cars_noverif", "cars"]:
            d = df[(df.attacker == a) & (df.policy == p)].sort_values("v")
            below = d[d.reduction < 2.0]
            row[f"v* ({POLICY_LABELS[p]})"] = f"{below.v.iloc[0]:g}" if len(below) else "> 0.5"
            row[f"Reduction at v=0 ({POLICY_LABELS[p]})"] = f"{d.reduction.iloc[0]:.1f}x"
        bnd.append(row)
    write_table(pd.DataFrame(bnd), "table_e1_boundary",
                "P1 boundary: smallest verification probability v* at which deception yields less than a 2x "
                "reduction in real compromises vs. blocking.", "tab:e1-boundary")

    # (c) cumulative compromise curves
    cfgs = [SimConfig(policy=p, attacker="UCB1", budget=S["budget"], v_att=0.05, seed=s)
            for p in ["block", "fixed_shaping", "cars_noverif", "cars"] for s in range(R)]
    rows = run_many(_run, cfgs, "E1c curves")
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for p, rs in _group(rows, ["policy"]).items():
        xs = [x for x, _ in rs[0]["curve"]]
        ys = np.array([[y for _, y in r["curve"]] for r in rs])
        m, sd = ys.mean(0), ys.std(0)
        ax.plot(xs, m, color=POLICY_COLORS[p[0]], label=POLICY_LABELS[p[0]], lw=1.2)
        ax.fill_between(xs, m - 1.96 * sd / math.sqrt(len(rs)), m + 1.96 * sd / math.sqrt(len(rs)),
                        color=POLICY_COLORS[p[0]], alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Attacker interactions"); ax.set_ylabel("Cumulative real compromises")
    ax.set_title("UCB1 attacker, v = 0.05")
    ax.legend(frameon=False)
    savefig(fig, "fig_e1c_curves")


def plot_e1b(df, atts):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, p in zip(axes, ["cars_noverif", "cars"]):
        for a in atts:
            d = df[(df.attacker == a) & (df.policy == p)]
            ax.plot([max(v, 0.002) for v in d.v], d.reduction, marker="o", ms=3, lw=1, label=a)
        ax.axhline(1, color="k", lw=0.7, ls=":")
        ax.axhline(2, color="grey", lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Verification probability v")
        ax.set_title(POLICY_LABELS[p])
    axes[0].set_ylabel("Reduction vs. blocking (x)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("P1: where verification ends deception's value (v = 0 plotted at 0.002)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "fig_e1b_boundary")


def replot():
    """Rebuild figures that can be drawn from saved results without re-running."""
    rows = load_jsonl("e1b_boundary")
    if rows:
        g = _group(rows, ["attacker", "v_att", "policy"])
        atts = list(dict.fromkeys(r["attacker"] for r in rows))
        vs_b = sorted({r["v_att"] for r in rows})
        recs = []
        for a in atts:
            for v in vs_b:
                cb = np.mean([r["C"] for r in g[(a, v, "block")]])
                for p in ["cars_noverif", "cars"]:
                    cp = np.mean([r["C"] for r in g[(a, v, p)]])
                    recs.append({"attacker": a, "v": v, "policy": p, "reduction": cb / max(cp, 1e-9)})
        plot_e1b(pd.DataFrame(recs), atts)
    rows = load_jsonl("e2_collective")
    if rows:
        plot_e2(_e2_df(rows), rows[0]["v_att"])


# ============================================================== E2 / P2
def e2(quick: bool):
    S = settings(quick)
    R = S["runs"]
    Ns = [1, 2, 4, 8]
    sharings = ["none", "exposure", "memory"]
    v = 0.05
    cfgs = []
    for pol in ["cars_minus_pop", "cars"]:
        for att in ["UCB1", "Thompson"]:
            for N in Ns:
                for sh in sharings:
                    if N == 1 and sh != "none":
                        continue
                    for s in range(R):
                        cfgs.append(SimConfig(policy=pol, attacker=att, n_attackers=N, sharing=sh,
                                              budget=S["budget"], v_att=v, seed=s))
    rows = run_many(_run, cfgs, "E2 budget-matched")
    for r in rows:
        r.pop("curve", None)
        firsts = [x for x in r["per_attacker_first_exposure"] if x is not None]
        r["frac_attackers_exposed"] = len(firsts) / r["N"]
    save_jsonl(rows, "e2_collective")

    g = _group(rows, ["policy", "attacker", "N", "sharing"])
    recs = []
    for (pol, att, N, sh), rs in sorted(g.items()):
        lt = mean_ci([r["decoy_pulls_before_decoy_exposure"] for r in rs])
        c = mean_ci([r["C"] for r in rs])
        fab = mean_ci([r["fab"] for r in rs])
        recs.append({"policy": pol, "attacker": att, "N": N, "sharing": sh,
                     "lifetime": lt, "C": c, "fab": fab,
                     "exposed_frac": float(np.mean([r["frac_attackers_exposed"] for r in rs]))})
    df = pd.DataFrame(recs)

    # P2 test: effect of sharing net of the budget (same total interactions)
    tests = []
    for pol in ["cars_minus_pop", "cars"]:
        for att in ["UCB1", "Thompson"]:
            for N in Ns[1:]:
                base = [r["C"] for r in g[(pol, att, N, "none")]]
                for sh in ["exposure", "memory"]:
                    other = [r["C"] for r in g[(pol, att, N, sh)]]
                    d, lo, hi = diff_ci(other, base)
                    tests.append({"Policy": POLICY_LABELS[pol], "Attacker": att, "N": N,
                                  "Sharing vs none": sh,
                                  "Delta real compromises [95% CI]": f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]",
                                  "Significant": "yes" if (lo > 0 or hi < 0) else "no"})
    write_table(pd.DataFrame(tests), "table_e2_sharing_effect",
                "P2: effect of attacker information sharing on real compromises at an equal total budget "
                f"(v = {v}). Positive = sharing helps the attacker.", "tab:e2-sharing")

    plot_e2(df, v)

    # rate effect: parallel schedule (every attacker acts each step)
    cfgs = [SimConfig(policy="cars_minus_pop", attacker="UCB1", n_attackers=N, sharing="exposure",
                      schedule="parallel", budget=S["budget"] // 4, v_att=v, seed=s)
            for N in Ns for s in range(R)]
    rows = run_many(_run, cfgs, "E2 parallel")
    recs = []
    for (N,), rs in sorted(_group(rows, ["N"]).items()):
        t = [r["decoy_first_exposure_t"] / N for r in rs if r["decoy_first_exposure_t"] is not None]
        recs.append({"N": N, "steps": mean_ci(t)})
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    m = np.array([r["steps"][0] for r in recs]); lo = np.array([r["steps"][1] for r in recs]); hi = np.array([r["steps"][2] for r in recs])
    ax.errorbar([r["N"] for r in recs], m, yerr=[m - lo, hi - m], marker="o", capsize=2, label="observed")
    ax.plot([r["N"] for r in recs], m[0] / np.array([r["N"] for r in recs]), "k--", lw=0.8, label="∝ 1/N")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Attackers N (acting in parallel)"); ax.set_ylabel("Wall-clock steps to 1st exposure")
    ax.set_title("Rate effect, not collective learning"); ax.legend(frameon=False)
    savefig(fig, "fig_e2_parallel_rate")


def _e2_df(rows):
    g = _group(rows, ["policy", "attacker", "N", "sharing"])
    recs = []
    for (pol, att, N, sh), rs in sorted(g.items()):
        recs.append({"policy": pol, "attacker": att, "N": N, "sharing": sh,
                     "lifetime": mean_ci([r["decoy_pulls_before_decoy_exposure"] for r in rs]),
                     "C": mean_ci([r["C"] for r in rs])})
    return pd.DataFrame(recs)


def plot_e2(df, v):
    sharings = ["none", "exposure", "memory"]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0))
    pred = predicted_lifetime(0.5, v, 1.0)
    for sh, mk in zip(sharings, ["o", "s", "^"]):
        d = df[(df.policy == "cars_minus_pop") & (df.attacker == "UCB1") & (df.sharing == sh)]
        if d.empty:
            continue
        m = np.array([x[0] for x in d.lifetime]); lo = np.array([x[1] for x in d.lifetime]); hi = np.array([x[2] for x in d.lifetime])
        axes[0].errorbar(d.N, m, yerr=[m - lo, hi - m], marker=mk, label=sh, lw=1, ms=3.5, capsize=2)
    axes[0].axhline(pred, color="k", ls="--", lw=0.8, label="1/h prediction")
    axes[0].set_ylim(0, None)
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("Attackers N")
    axes[0].set_ylabel("Decoy interactions to 1st exposure")
    axes[0].set_title("(a) First exposure: budget-driven", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    for ax, pol, ttl in [(axes[1], "cars_minus_pop", "(b) Shared decoy identity"),
                         (axes[2], "cars", "(c) Session-unique decoys")]:
        for sh, mk in zip(sharings, ["o", "s", "^"]):
            d = df[(df.policy == pol) & (df.attacker == "UCB1") & (df.sharing == sh)]
            if d.empty:
                continue
            m = np.array([x[0] for x in d.C]); lo = np.array([x[1] for x in d.C]); hi = np.array([x[2] for x in d.C])
            ax.errorbar(d.N, m, yerr=[m - lo, hi - m], marker=mk, label=sh, lw=1, ms=3.5, capsize=2)
        ax.set_xscale("log", base=2); ax.set_xlabel("Attackers N"); ax.set_title(ttl, fontsize=9)
        ax.set_ylabel("Real compromises (equal budget)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"P2: collective verification at an equal total budget (UCB1, v = {v})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, "fig_e2_collective")


# ============================================================== E3s / P3
def e3_synthetic(quick: bool):
    S = settings(quick)
    R = S["runs"]
    atts = ATTACKERS_ALL if not quick else ["UCB1", "Suspicious-UCB", "EXP3"]
    scenarios = [("single, v=0", 1, "none", 0.0), ("single, v=0.05", 1, "none", 0.05),
                 ("4 sharing, v=0.05", 4, "memory", 0.05)]
    cfgs = [SimConfig(policy=p, attacker=a, n_attackers=N, sharing=sh, budget=S["budget"], v_att=v, seed=s)
            for (_, N, sh, v) in scenarios for a in atts for p in MAIN_POLICIES for s in range(R)]
    rows = run_many(_run, cfgs, "E3 synthetic")
    for r in rows:
        r.pop("curve", None)
    save_jsonl(rows, "e3_synthetic")
    g = _group(rows, ["N", "sharing", "v_att", "attacker", "policy"])

    for (label, N, sh, v) in scenarios:
        tab = []
        for a in atts:
            row = {"Attacker": a}
            for p in MAIN_POLICIES:
                row[POLICY_LABELS[p]] = fmt_ci(*mean_ci([r["C"] for r in g[(N, sh, v, a, p)]]))
            tab.append(row)
        name = "table_e3_" + label.replace(" ", "").replace(",", "_").replace("=", "").replace(".", "")
        write_table(pd.DataFrame(tab), name,
                    f"P3 (synthetic, {label}): real compromises, mean [95% CI], equal budget of {S['budget']} "
                    "interactions. *Simplified re-implementation of the state-consistent adaptive idea, "
                    "not equivalent to AgentSnare.", f"tab:{name}")

    summ = []
    for (label, N, sh, v) in scenarios:
        for p in MAIN_POLICIES:
            rs = [r for a in atts for r in g[(N, sh, v, a, p)]]
            summ.append({"Scenario": label, "Policy": POLICY_LABELS[p],
                         "C": fmt_ci(*mean_ci([r["C"] for r in rs])),
                         "I (intel)": fmt_ci(*mean_ci([r["I"] for r in rs])),
                         "kappa": fmt_ci(*mean_ci([r["kappa"] for r in rs]), digits=1),
                         "J": fmt_ci(*mean_ci([r["J"] for r in rs]), digits=1),
                         "Rotations": f"{np.mean([r['rotations'] for r in rs]):.1f}"})
    write_table(pd.DataFrame(summ), "table_e3_objective",
                "P3 (synthetic): objective terms averaged over attackers. J = C + lambda*kappa - eta*I "
                "(lambda=1, eta=0.01).", "tab:e3-objective")

    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.4, 3.0), sharey=True)
    w = 0.13
    for ax, (label, N, sh, v) in zip(axes, scenarios):
        x = np.arange(len(atts))
        for k, p in enumerate(MAIN_POLICIES):
            ms = [mean_ci([r["C"] for r in g[(N, sh, v, a, p)]]) for a in atts]
            m = np.array([max(t[0], 0.5) for t in ms])
            ax.bar(x + (k - 2.5) * w, m, w, color=POLICY_COLORS[p], label=POLICY_LABELS[p])
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(atts, rotation=45, ha="right", fontsize=7)
        ax.set_title(label)
    axes[0].set_ylabel("Real compromises (log)")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    savefig(fig, "fig_e3_synthetic_policies")


# ============================================================== E4 ablation
def e4(quick: bool):
    S = settings(quick)
    R = S["runs"]
    atts = ["UCB1", "Suspicious-UCB", "Thompson"]
    # semantic scenario: templated fabrications self-expose with probability 0.003 each
    cfgs = [SimConfig(policy=p, attacker=a, n_attackers=4, sharing="memory", budget=S["budget"],
                      v_att=0.05, template_inconsistency=0.003, seed=s)
            for a in atts for p in ABLATIONS for s in range(R)]
    rows = run_many(_run, cfgs, "E4 ablation")
    for r in rows:
        r.pop("curve", None)
    save_jsonl(rows, "e4_ablation")
    g = _group(rows, ["attacker", "policy"])
    tab = []
    for a in atts:
        base = [r["C"] for r in g[(a, "cars")]]
        for p in ABLATIONS:
            cs = [r["C"] for r in g[(a, p)]]
            d, lo, hi = diff_ci(cs, base) if p != "cars" else (0, 0, 0)
            tab.append({"Attacker": a, "Variant": POLICY_LABELS[p], "Real compromises": fmt_ci(*mean_ci(cs)),
                        "Delta vs full [95% CI]": "--" if p == "cars" else f"{d:+.0f} [{lo:+.0f}, {hi:+.0f}]",
                        "Dimension needed": "--" if p == "cars" else ("yes" if lo > 0 else "no")})
    write_table(pd.DataFrame(tab), "table_e4_ablation",
                "E4: removing one credibility dimension at a time (4 sharing attackers, v = 0.05, templated "
                "inconsistency 0.003). A dimension is kept only if removing it significantly increases "
                "compromises.", "tab:e4-ablation")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    x = np.arange(len(atts)); w = 0.16
    for k, p in enumerate(ABLATIONS):
        ms = [mean_ci([r["C"] for r in g[(a, p)]]) for a in atts]
        m = np.array([t[0] for t in ms]); lo = np.array([t[1] for t in ms]); hi = np.array([t[2] for t in ms])
        ax.bar(x + (k - 2) * w, m, w, yerr=[m - lo, hi - m], capsize=1.5, color=POLICY_COLORS[p],
               label=POLICY_LABELS[p], error_kw={"lw": 0.6})
    ax.set_xticks(x); ax.set_xticklabels(atts)
    ax.set_ylabel("Real compromises"); ax.set_title("E4: ablation of credibility dimensions")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    savefig(fig, "fig_e4_ablation")


EXPS = {"e1": e1, "e2": e2, "e3s": e3_synthetic, "e4": e4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", *EXPS])
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    ap.add_argument("--replot", action="store_true", help="redraw figures from saved results only")
    a = ap.parse_args()
    if a.replot:
        replot()
        return
    for name, fn in EXPS.items():
        if a.exp in ("all", name):
            print(f"=== {name} ===", flush=True)
            fn(a.quick)


if __name__ == "__main__":
    main()
