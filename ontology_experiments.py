"""E7: does an ontology help credibility-aware reward shaping?

E7a  knowledge quality: verifiability, ATLAS technique and the fabricate/block decision for
     tools the defender never modelled. Flat lookup table vs. ontology reasoning, on
     (i) AgentDojo suites outside the modelled systems (slack, travel) and
     (ii) 30 held-out tool names from other ecosystems (labels provisional, see README).
E7b  downstream effect (synthetic): real avenues get the verifiability of randomly drawn
     held-out tools; CARS runs with a flat KB, ontology estimates only, ontology estimates +
     ontology-guided decoy families, or oracle knowledge. Sweeps verification probability v
     and family-level generalisation g. v = 0 is a control where no difference is expected.

    python -m experiments.ontology_experiments [--quick]
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cars.live.knowledge import MODELLED_SUITES, TOOL_KB
from cars.ontology.reasoner import decoy_families, infer
from cars.synthetic.policies import PRESETS
from cars.synthetic.sim import DEFAULT_ARMS, Arm, SimConfig, run

from .common import diff_ci, fmt_ci, mean_ci, plt, run_many, save_jsonl, savefig, settings, write_table

HELDOUT = Path(__file__).resolve().parents[1] / "cars" / "ontology" / "heldout_tools.csv"
BLOCK_VB = 0.7
PRESETS["cars_onto_est_only"] = replace(PRESETS["cars_onto"], name="cars_onto_est_only", family_aware=False)
E7_POLICIES = ["cars_flat_kb", "cars_onto_est_only", "cars_onto", "cars_oracle_kb"]
E7_LABELS = {"cars_flat_kb": "Flat KB", "cars_onto_est_only": "Ontology (estimates)",
             "cars_onto": "Ontology (estimates + decoy families)", "cars_oracle_kb": "Oracle knowledge"}
E7_COLORS = {"cars_flat_kb": "#9e9e9e", "cars_onto_est_only": "#4c9be8", "cars_onto": "#1f4e9c",
             "cars_oracle_kb": "#2ca02c"}


def heldout() -> list[dict]:
    return [{"tool": r["tool"], "verifiability": float(r["verifiability"]), "atlas": r["atlas"],
             "domain": r["domain"]} for r in csv.DictReader(HELDOUT.open())]


# ================================================================== E7a
def e7a():
    rows = []
    # (i) AgentDojo tools of systems outside the modelled ones: deployment mode (assertions used)
    for su, d in TOOL_KB.items():
        if su in MODELLED_SUITES:
            continue
        for t, (v, a) in d.items():
            o = infer(t, use_assertions=True)
            rows.append({"set": "AgentDojo unmodelled (slack, travel)", "tool": f"{su}:{t}", "truth_v": v,
                         "truth_atlas": a, "flat_v": 0.5, "flat_atlas": "AML.T0053", "onto_v": o["verifiability"],
                         "onto_atlas": o["atlas"], "onto_class": o["class"], "onto_how": o["how"]})
    # (ii) held-out tool names, lexical inference only
    for r in heldout():
        o = infer(r["tool"], use_assertions=False)
        rows.append({"set": "Held-out tools (30, other ecosystems)", "tool": r["tool"], "truth_v": r["verifiability"],
                     "truth_atlas": r["atlas"], "flat_v": 0.5, "flat_atlas": "AML.T0053",
                     "onto_v": o["verifiability"], "onto_atlas": o["atlas"], "onto_class": o["class"],
                     "onto_how": o["how"]})
    df = pd.DataFrame(rows)
    df.to_csv(Path("tables") / "table_e7a_per_tool.csv", index=False)
    summ = []
    for st, d in df.groupby("set", sort=False):
        for kb in ("flat", "onto"):
            dec_true = d.truth_v >= BLOCK_VB
            dec = d[f"{kb}_v"] >= BLOCK_VB
            summ.append({"Tool set": st, "Knowledge": "Flat KB" if kb == "flat" else "Ontology",
                         "Tools": len(d), "Verifiability MAE": f"{(d[f'{kb}_v'] - d.truth_v).abs().mean():.3f}",
                         "ATLAS accuracy %": f"{100 * (d[f'{kb}_atlas'] == d.truth_atlas).mean():.0f}",
                         "Fabricate/block decision agreement %": f"{100 * (dec == dec_true).mean():.0f}"})
    write_table(pd.DataFrame(summ), "table_e7a_knowledge",
                "E7a: knowledge for tools the defender never modelled. Flat lookup table (unknown tool -> "
                f"verifiability 0.5, generic ATLAS label) vs. ontology reasoning. Decision = block fabrication when "
                f"verifiability >= {BLOCK_VB}. Held-out labels are provisional (authors) until expert annotation.",
                "tab:e7a")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    sets = list(dict.fromkeys(df.set))
    x = np.arange(len(sets)); w = 0.35
    for k, kb in enumerate(("flat", "onto")):
        mae = [(df[df.set == s][f"{kb}_v"] - df[df.set == s].truth_v).abs().mean() for s in sets]
        acc = [100 * (df[df.set == s][f"{kb}_atlas"] == df[df.set == s].truth_atlas).mean() for s in sets]
        lab = "Flat KB" if kb == "flat" else "Ontology"
        col = "#9e9e9e" if kb == "flat" else "#1f4e9c"
        axes[0].bar(x + (k - 0.5) * w, mae, w, label=lab, color=col)
        axes[1].bar(x + (k - 0.5) * w, acc, w, label=lab, color=col)
    for ax, t in zip(axes, ["Verifiability error (MAE, lower is better)", "ATLAS label accuracy %"]):
        ax.set_xticks(x); ax.set_xticklabels(["AgentDojo\nunmodelled", "Held-out\ntools"]); ax.set_title(t, fontsize=9)
    axes[0].legend(frameon=False)
    savefig(fig, "fig_e7a_knowledge")
    return df


# ================================================================== E7b
def _cfg(policy: str, seed: int, v: float, g: float, attacker: str, budget: int) -> SimConfig:
    tools = heldout()
    pick = random.Random(seed).sample(tools, len(DEFAULT_ARMS))
    arms = [Arm(t["tool"], a.mu, a.rho, t["verifiability"], t["atlas"]) for a, t in zip(DEFAULT_ARMS, pick)]
    if policy == "cars_flat_kb":
        est = [0.5] * len(arms)
    elif policy == "cars_oracle_kb":
        est = None
    else:
        est = [infer(t["tool"], use_assertions=False)["verifiability"] for t in pick]
    return SimConfig(policy_override=PRESETS[policy], attacker=attacker, n_attackers=4, sharing="memory",
                     budget=budget, v_att=v, decoy_families=decoy_families(), g_family=g, vb_est=est,
                     arms=arms, seed=seed)


def _run(c):
    r = run(c)
    r.pop("curve", None)
    r["g_family"] = c.g_family
    return r


def e7b(quick: bool):
    S = settings(quick)
    R = S["runs"]
    vs = [0.0, 0.05, 0.1, 0.3]
    gs = [0.0, 0.5]
    atts = ["UCB1", "Skeptical-UCB", "Thompson"] if not quick else ["UCB1"]
    budget = S["budget_small"] if quick else 10000
    cfgs = [_cfg(p, s, v, g, a, budget) for a in atts for v in vs for g in gs for p in E7_POLICIES for s in range(R)]
    rows = run_many(_run, cfgs, "E7b ontology downstream")
    save_jsonl(rows, "e7b_ontology")
    g = defaultdict(list)
    for r in rows:
        g[(r["attacker"], r["v_att"], r["g_family"], r["policy"])].append(r["C"])
    tab = []
    for a in atts:
        for v in vs:
            for gf in gs:
                base = g[(a, v, gf, "cars_flat_kb")]
                for p in E7_POLICIES:
                    cs = g[(a, v, gf, p)]
                    d, lo, hi = diff_ci(cs, base) if p != "cars_flat_kb" else (0, 0, 0)
                    tab.append({"Attacker": a, "v": v, "g_family": gf, "Knowledge": E7_LABELS[p],
                                "Real compromises": fmt_ci(*mean_ci(cs)),
                                "Delta vs flat KB [95% CI]": "--" if p == "cars_flat_kb" else f"{d:+.0f} [{lo:+.0f}, {hi:+.0f}]",
                                "Better than flat": "--" if p == "cars_flat_kb" else ("yes" if hi < 0 else ("worse" if lo > 0 else "n.s."))})
    write_table(pd.DataFrame(tab), "table_e7b_downstream",
                "E7b: effect of the knowledge layer on real compromises (4 sharing attackers, equal budget). "
                "Real avenues take the verifiability of randomly drawn held-out tools. v = 0 is a control.",
                "tab:e7b")
    fig, axes = plt.subplots(1, len(gs), figsize=(3.4 * len(gs), 2.8), sharey=True, squeeze=False)
    for ax, gf in zip(axes[0], gs):
        x = np.arange(len(vs)); w = 0.2
        for k, p in enumerate(E7_POLICIES):
            ms = [mean_ci([c for a in atts for c in g[(a, v, gf, p)]]) for v in vs]
            m = np.array([t[0] for t in ms]); lo = np.array([t[1] for t in ms]); hi = np.array([t[2] for t in ms])
            ax.bar(x + (k - 1.5) * w, m, w, yerr=[m - lo, hi - m], capsize=1.5, color=E7_COLORS[p],
                   label=E7_LABELS[p], error_kw={"lw": 0.6})
        ax.set_xticks(x); ax.set_xticklabels([f"v={v}" for v in vs])
        ax.set_title(f"Family-level generalisation g = {gf}")
    axes[0][0].set_ylabel("Real compromises (mean over attackers)")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("E7b: does the ontology help reward shaping?", y=1.02, fontsize=10)
    savefig(fig, "fig_e7b_downstream")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print("=== e7 (ontology) ===")
    e7a()
    e7b(a.quick)


if __name__ == "__main__":
    main()
