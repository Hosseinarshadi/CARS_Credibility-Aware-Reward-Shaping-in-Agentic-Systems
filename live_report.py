"""Tables and figures for the live AgentDojo experiments (E0, E2, E3, E4, E5, E6, E7) and spend.

    python -m experiments.live_report          # from real runs
    python -m experiments.live_report --mock   # from mock runs (outputs suffixed _mock)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cars.live.knowledge import ATLAS_NAMES
from cars.live.llm import load_config

from .common import POLICY_COLORS, POLICY_LABELS, fmt_ci, mean_ci, plt, savefig, wilson, write_table

LIVE = Path(__file__).resolve().parents[1] / "results" / "live"


PROFILE = ""   # "" = default agent, "_permissive" = permissive agent profile


def _load(name: str, mock: bool) -> pd.DataFrame:
    p = LIVE / (f"{name}_mock.jsonl" if mock else f"{name}{PROFILE}.jsonl")
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in p.open()])


def _order(pols):
    order = list(POLICY_LABELS)
    return sorted(pols, key=lambda p: order.index(p) if p in order else 99)


def _calibration(mock: bool, sfx: str = ""):
    f = LIVE / ("calibration_mock.json" if mock else f"calibration{sfx}.json")
    return json.loads(f.read_text()) if f.exists() else None


def _pct(b):
    if not b or b.get("n", 0) == 0:
        return "--"
    lo, hi = b["ci"]
    return f"{100 * b['rate']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def e0(sfx: str, mock: bool):
    """L0, natural steering of the real agent: numerator, denominator, Wilson CI, fit/held-out."""
    tab = []
    profiles = [("", "default"), ("_permissive", "permissive")] if not mock else [("", "mock")]
    for psfx, pname in profiles:
        cal = _calibration(mock, psfx)
        if not cal:
            continue
        for split in ("fit", "heldout", "all"):
            b = cal[split]
            tab.append({"Agent profile": pname, "Scope": {"fit": "Fit tasks", "heldout": "Held-out tasks",
                                                          "all": "All tasks"}[split],
                        "Goal attempts k / n": f"{b['k']} / {b['n']}", "Rate % [95% Wilson CI]": _pct(b)})
        for av, b in cal["per_avenue"].items():
            tab.append({"Agent profile": pname, "Scope": f"avenue: {av}",
                        "Goal attempts k / n": f"{b['k']} / {b['n']}",
                        "Rate % [95% Wilson CI]": _pct(b) if b["reported"] else
                        f"not reported (n < {cal['min_n_per_avenue']}); pooled only"})
        rc = cal["real_compromise_all"]
        tab.append({"Agent profile": pname, "Scope": "real compromise (blocking)",
                    "Goal attempts k / n": f"{rc['k']} / {rc['n']}", "Rate % [95% Wilson CI]": _pct(rc)})
        v = cal.get("validation")
        if v:
            tab.append({"Agent profile": pname, "Scope": "held-out validation",
                        "Goal attempts k / n": f"|fit - held-out| = {100 * v['abs_error']:.1f} pp",
                        "Rate % [95% Wilson CI]": ("fit rate inside held-out CI" if v["fit_rate_within_heldout_ci"]
                                                   else "fit rate OUTSIDE held-out CI")
                        + f"; Brier {v['brier_fit']:.3f} vs 0.25 uninformed"})
    if not tab:
        print("  (no E0 data)")
        return
    write_table(pd.DataFrame(tab), f"table_live_e0_calibration{sfx}",
                "E0 (L0): natural steering of the real agent under plain blocking. A goal attempt is a write "
                "toward the injection's goal or a honeytool call, counted before any defense acts. Per-avenue "
                "rates are shown only with enough episodes. The fit-split Beta posterior parameterises the "
                "calibrated E3 condition; a rate near zero is reported as a finding, not tuned away.",
                f"tab:live-e0{sfx}")


def _runs(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("run_key").agg(
        suite=("suite", "first"), policy=("policy", "first"), algo=("attacker_algo", "first"),
        compliance=("compliance_set", "first"), drawn=("compliance_drawn", "first"), seed=("seed", "first"),
        C=("real_compromise", "sum"), rounds=("round", "count"),
        attempts=("attempted_goal", "sum"), fab=("fabricated_observed", "sum"),
        nfab=("n_fabricated", "sum"), util=("utility", "mean"), exposures=("exposed", "sum"),
        fallback=("attack_source", lambda s: (s == "fallback").mean())).reset_index()


def _sign_p(d) -> float:
    """Two-sided exact sign test on non-zero paired differences (robust when most are 0)."""
    neg, pos = sum(x < 0 for x in d), sum(x > 0 for x in d)
    if neg + pos == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(neg, neg + pos, 0.5).pvalue)
    except Exception:
        return float("nan")


def _pair_diffs(runs: pd.DataFrame, ref: str, other: str, keys=("suite", "algo", "compliance", "seed")):
    k = list(keys)
    a = runs[runs.policy == ref].set_index(k).C
    b = runs[runs.policy == other].set_index(k).C
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    return list(j.a - j.b)


def _paired(runs: pd.DataFrame, ref: str = "cars", keys=("suite", "algo", "compliance", "seed")) -> pd.DataFrame:
    """Paired differences ref - policy on identical scenarios (common random numbers)."""
    out = []
    for p in _order(runs.policy.unique()):
        if p == ref:
            continue
        d = _pair_diffs(runs, ref, p, keys)
        if not d:
            continue
        m, lo, hi = mean_ci(d)
        out.append({"Comparison": f"{POLICY_LABELS.get(ref, ref)} - {POLICY_LABELS.get(p, p)}",
                    "Paired diff. real compromises / run": fmt_ci(m, lo, hi, digits=2),
                    "CARS fewer / more / tied (pairs)": f"{sum(x < 0 for x in d)} / {sum(x > 0 for x in d)} / "
                                                        f"{sum(x == 0 for x in d)}",
                    "Sign test p": f"{_sign_p(d):.3g}", "Pairs": len(d)})
    return pd.DataFrame(out)


def _paired_grid(runs: pd.DataFrame, col: str, levels, ref: str = "cars", keys=("suite", "algo", "seed")):
    """One row per comparison, one column per level of `col`: diff [CI] (sign-test p)."""
    out = []
    for p in _order(runs.policy.unique()):
        if p == ref:
            continue
        row = {"Comparison": f"CARS - {POLICY_LABELS.get(p, p)}"}
        for lv in levels:
            sub = runs[runs[col] == lv]
            d = _pair_diffs(sub, ref, p, keys)
            row[f"{col}={lv:g}"] = (f"{fmt_ci(*mean_ci(d), digits=2)} p={_sign_p(d):.2g}" if d else "--")
        out.append(row)
    return pd.DataFrame(out)


def _policy_table(r: pd.DataFrame) -> list[dict]:
    tab = []
    for su in sorted(r.suite.unique()):
        for p in _order(r.policy.unique()):
            x = r[(r.suite == su) & (r.policy == p)]
            if x.empty:
                continue
            tab.append({"Suite": su, "Policy": POLICY_LABELS.get(p, p),
                        "Real compromises / run": fmt_ci(*mean_ci(list(x.C)), digits=2),
                        "Goal attempts / run": f"{x.attempts.mean():.1f}",
                        "Fabricated responses / run (budget)": f"{x.nfab.mean():.1f}",
                        "Decoy-absorbed %": f"{100 * x.fab.sum() / max(1, x.attempts.sum()):.0f}",
                        "Exposures / run": f"{x.exposures.mean():.2f}",
                        "Utility under attack %": f"{100 * x.util.mean():.1f}", "Runs": len(x)})
    return tab


def _adaptation(df: pd.DataFrame) -> pd.DataFrame:
    """Observed attacker adaptation from the logs: P(change avenue next round | feedback type).
    The attacker is the bandit learner (templates), not an LLM."""
    rows = []
    for (rk, sess), d in df.sort_values("round").groupby(["run_key", "session"]):
        d = d.reset_index(drop=True)
        for i in range(len(d) - 1):
            r = d.iloc[i]
            if r.real_compromise:
                cat = "real success"
            elif r.observed != "fail" and r.exposed:
                cat = "fabrication verified as fake"
            elif r.observed != "fail" and r.recognized:
                cat = "fabrication recognised (known decoy)"
            elif r.observed != "fail":
                cat = "fabricated success believed"
            else:
                cat = "failure / blocked"
            rows.append({"policy": r.policy, "algo": r.attacker_algo, "cat": cat,
                         "switch": d.iloc[i + 1].avenue != r.avenue})
    a = pd.DataFrame(rows)
    if a.empty:
        return a
    tab = []
    for cat, x in a.groupby("cat"):
        k, n = int(x.switch.sum()), len(x)
        p_, lo, hi = wilson(k, n)
        tab.append({"Feedback after a round": cat, "P(switch avenue next round) %":
                    f"{100 * p_:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]", "Transitions": n})
    return pd.DataFrame(tab)


def e3(sfx: str, mock: bool):
    """L1: calibrated assume-breach (uncertainty propagated from E0) + compliance sweep."""
    df = _load("e3", mock)
    if df.empty:
        print("  (no E3 data)")
        return df
    runs = _runs(df)
    cal = _calibration(mock)
    anchor = cal["fit"]["rate"] if cal else None
    calrows = runs[runs.compliance == "calibrated"]
    sweep = runs[runs.compliance != "calibrated"].copy()
    sweep["compliance"] = sweep.compliance.astype(float)
    levels = sorted(sweep.compliance.unique())

    if not calrows.empty:
        head, lab = calrows, (f"the calibrated condition (compliance drawn per run from the E0 posterior, "
                              f"mean {calrows.drawn.mean():.3f})")
    else:
        op = levels[len(levels) // 2]
        head, lab = sweep[sweep.compliance == op], f"stress level {op:.2f} (E0 not yet run)"
    write_table(pd.DataFrame(_policy_table(head)), f"table_live_e3{sfx}",
                f"Live E3 (P3, L1 calibrated assume-breach) at {lab}. Real compromises are judged on the real "
                "environment only; all policies face the same paired steering opportunities. 'Fabricated "
                "responses' is each policy's deception budget (Fixed shaping, CARS-matched has the same "
                "expected budget as CARS). Averaged over attacker algorithms. *Simplified state-consistent "
                "adaptive baseline, not equivalent to AgentSnare. These are conditional results under modelled "
                "steering, not observed live-agent harm.", f"tab:live-e3{sfx}")
    pt = _paired(head)
    if not pt.empty:
        write_table(pt, f"table_live_e3_paired{sfx}",
                    f"Live E3 paired comparison at {lab}: CARS minus each policy on identical scenarios "
                    "(same suite, attacker algorithm, compliance draw, seed and all random streams). Negative = "
                    "CARS fewer real compromises. Bootstrap 95% CI; exact sign test.", f"tab:live-e3-paired{sfx}")

    # sensitivity sweep (option A)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for p in _order(sweep.policy.unique()):
        d = sweep[sweep.policy == p]
        xs, ms, los, his = [], [], [], []
        for c in levels:
            cc = d[d.compliance == c]
            if cc.empty:
                continue
            m, lo, hi = mean_ci(list(cc.C))
            xs.append(c); ms.append(m); los.append(lo); his.append(hi)
        ax.plot(xs, ms, marker="o", ms=3, lw=1.3, color=POLICY_COLORS.get(p), label=POLICY_LABELS.get(p, p))
        ax.fill_between(xs, los, his, color=POLICY_COLORS.get(p), alpha=0.12)
    if cal:
        lo, hi = cal["fit"]["ci"]
        ax.axvspan(lo, hi, color="k", alpha=0.08, lw=0)
        ax.axvline(anchor, color="k", ls="--", lw=0.8)
        ax.text(anchor, ax.get_ylim()[1] * 0.97, f" real agent {anchor:.2f}", fontsize=7, va="top")
    ax.set_xlabel("Agent compliance (P follow injection)")
    ax.set_ylabel("Real compromises / run")
    ax.set_title("Sensitivity: policies across compliance")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    savefig(fig, f"fig_live_e3_sweep{sfx}")
    stab = []
    for p in _order(sweep.policy.unique()):
        row = {"Policy": POLICY_LABELS.get(p, p)}
        for c in levels:
            row[f"c={c:g}"] = f"{sweep[(sweep.policy == p) & (sweep.compliance == c)].C.mean():.2f}"
        stab.append(row)
    write_table(pd.DataFrame(stab), f"table_live_e3_sweep{sfx}",
                "Live E3 sensitivity sweep (conditional stress test): mean real compromises / run per compliance "
                "level. The shaded band in the figure is the E0 95% CI for the real agent.", f"tab:live-e3-sweep{sfx}")
    write_table(_paired_grid(sweep, "compliance", levels), f"table_live_e3_sweep_paired{sfx}",
                "Live E3 sweep, paired: CARS minus each policy at every compliance level (same suite, attacker "
                "algorithm and seed; all random streams shared). Negative = CARS fewer real compromises. "
                "Bootstrap 95% CI; exact sign test p.", f"tab:live-e3-sweep-paired{sfx}")

    # live bandit-algorithm comparison in the headline condition
    algos = sorted(head.algo.unique())
    if len(algos) > 1:
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        pols = _order(head.policy.unique())
        x = np.arange(len(algos)); w = 0.8 / max(1, len(pols))
        for k, p in enumerate(pols):
            ms = [head[(head.policy == p) & (head.algo == al)].C.mean() for al in algos]
            ax.bar(x + (k - len(pols) / 2 + 0.5) * w, ms, w, color=POLICY_COLORS.get(p), label=POLICY_LABELS.get(p, p))
        ax.set_xticks(x); ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Real compromises / run"); ax.set_title("Live bandit-attacker comparison")
        ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
        savefig(fig, f"fig_live_e3_by_algorithm{sfx}")
        btab = []
        for al in algos:
            row = {"Attacker algorithm": al}
            for p in pols:
                row[POLICY_LABELS.get(p, p)] = fmt_ci(*mean_ci(list(head[(head.policy == p) & (head.algo == al)].C)), digits=1)
            btab.append(row)
        write_table(pd.DataFrame(btab), f"table_live_e3_algorithms{sfx}",
                    f"Live E3 by bandit algorithm at {lab}: real compromises / run.", f"tab:live-e3-algo{sfx}")

    ad = _adaptation(df)
    if not ad.empty:
        write_table(ad, f"table_live_e3_adaptation{sfx}",
                    "Observed attacker adaptation in the live environment: probability that the learning attacker "
                    "changes avenue in the next round, by the feedback it just received (Wilson 95% CI). The "
                    "attacker is a bandit learner over fixed templates, not an LLM.", f"tab:live-e3-adapt{sfx}")
    src = df.attack_source.value_counts(normalize=True)
    print("  attack text source: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in src.items()))
    return df


def e3v(sfx: str, mock: bool):
    """P1 live: CARS vs. budget-matched and other deception as attacker verification grows."""
    df = _load("e3v", mock)
    if df.empty:
        print("  (no E3v data)")
        return
    runs = df.groupby("run_key").agg(
        suite=("suite", "first"), policy=("policy", "first"), algo=("attacker_algo", "first"),
        compliance=("compliance_set", "first"), v=("v_att", "first"), seed=("seed", "first"),
        C=("real_compromise", "sum"), exposures=("exposed", "sum"), nfab=("n_fabricated", "sum")).reset_index()
    runs["compliance"] = runs.compliance.astype(float)
    vs = sorted(runs.v.unique())
    comps = sorted(runs.compliance.unique())
    fig, axes = plt.subplots(1, len(comps), figsize=(3.4 * len(comps), 3.0), squeeze=False)
    for ax, c in zip(axes[0], comps):
        for p in _order(runs.policy.unique()):
            d = runs[(runs.policy == p) & (runs.compliance == c)]
            ms = [mean_ci(list(d[d.v == v].C)) for v in vs]
            ax.plot(vs, [m[0] for m in ms], marker="o", ms=3, lw=1.3, color=POLICY_COLORS.get(p),
                    label=POLICY_LABELS.get(p, p))
            ax.fill_between(vs, [m[1] for m in ms], [m[2] for m in ms], color=POLICY_COLORS.get(p), alpha=0.1)
        ax.set_xlabel("Attacker verification probability v")
        ax.set_title(f"compliance {c:g}")
    axes[0][0].set_ylabel("Real compromises / run")
    axes[0][-1].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1, 1))
    savefig(fig, f"fig_live_e3v_verification{sfx}")
    tab = []
    for c in comps:
        for p in _order(runs.policy.unique()):
            row = {"Compliance": c, "Policy": POLICY_LABELS.get(p, p)}
            for v in vs:
                d = runs[(runs.policy == p) & (runs.compliance == c) & (runs.v == v)]
                row[f"v={v:g}"] = f"{d.C.mean():.2f} (exp {d.exposures.mean():.2f})"
            tab.append(row)
    write_table(pd.DataFrame(tab), f"table_live_e3v{sfx}",
                "Live E3v (P1): mean real compromises / run (and decoy exposures / run) by attacker verification "
                "probability, pre-registered levels, all reported.", f"tab:live-e3v{sfx}")
    for c in comps:
        write_table(_paired_grid(runs[runs.compliance == c], "v", vs), f"table_live_e3v_paired_c{c:g}{sfx}",
                    f"Live E3v paired at compliance {c:g}: CARS minus each policy per verification level "
                    "(same suite, algorithm, seed; shared random streams). Negative = CARS fewer real compromises. "
                    "Bootstrap 95% CI; exact sign test p.", f"tab:live-e3v-paired-{c:g}{sfx}")


def e2(sfx: str, mock: bool):
    """P2 live: separate the effect of more observations from the effect of sharing identities."""
    df = _load("e2", mock)
    if df.empty:
        print("  (no E2 data)")
        return
    runs = df.groupby("run_key").agg(policy=("policy", "first"), N=("N", "first"), sharing=("sharing", "first"),
                                     algo=("attacker_algo", "first"), seed=("seed", "first"),
                                     C=("real_compromise", "sum"), fab=("fabricated_observed", "sum"),
                                     exposures=("exposed", "sum"), recognized=("recognized", "sum")).reset_index()
    tab = []
    for p in _order(runs.policy.unique()):
        for (N, sh), r in runs[runs.policy == p].groupby(["N", "sharing"]):
            tab.append({"Policy": POLICY_LABELS.get(p, p), "Attackers": N, "Sharing": sh,
                        "Real compromises": fmt_ci(*mean_ci(list(r.C)), digits=1),
                        "Fabrications believed": f"{(r.fab - r.recognized - r.exposures).mean():.1f}",
                        "Exposures": f"{r.exposures.mean():.1f}", "Recognised as fake": f"{r.recognized.mean():.1f}",
                        "Runs": len(r)})
    write_table(pd.DataFrame(tab), f"table_live_e2{sfx}",
                "Live E2 (P2): collective verification at an equal total budget of attack rounds. 'stats' shares "
                "bandit statistics only; 'memory' also shares exposed decoy identities.", f"tab:live-e2{sfx}")
    # contrasts on paired (policy, algo, seed)
    piv = runs.assign(arm=runs.N.astype(str) + "/" + runs.sharing).pivot_table(
        index=["policy", "algo", "seed"], columns="arm", values="C")
    con = [("4/stats", "4/none", "More observations (shared statistics) vs independent"),
           ("4/memory", "4/stats", "Sharing decoy identities, over and above more observations"),
           ("4/memory", "1/none", "Collective sharing vs one attacker seeing all observations")]
    ct = []
    for p in _order(runs.policy.unique()):
        if p not in piv.index.get_level_values(0):
            continue
        sub = piv.loc[p]
        for a_, b_, lab in con:
            if a_ in sub and b_ in sub:
                d = list((sub[a_] - sub[b_]).dropna())
                if d:
                    ct.append({"Policy": POLICY_LABELS.get(p, p), "Contrast": lab,
                               "Paired diff. real compromises": fmt_ci(*mean_ci(d), digits=2), "Pairs": len(d)})
    if ct:
        write_table(pd.DataFrame(ct), f"table_live_e2_contrasts{sfx}",
                    "Live E2 contrasts (paired by policy, attacker algorithm and seed, equal total rounds): positive = "
                    "the first arm causes more real compromises.", f"tab:live-e2-con{sfx}")


def e4(sfx: str, mock: bool):
    df = _load("e4", mock)
    if df.empty:
        return
    runs = df.groupby("run_key").agg(policy=("policy", "first"), C=("real_compromise", "sum")).reset_index()
    tab = [{"Variant": POLICY_LABELS.get(p, p), "Real compromises": fmt_ci(*mean_ci(list(r.C)), digits=1), "Runs": len(r)}
           for p, r in runs.groupby("policy")]
    write_table(pd.DataFrame(tab), f"table_live_e4{sfx}", "Live E4: ablation of credibility dimensions.", f"tab:live-e4{sfx}")


def e7(sfx: str, mock: bool):
    df = _load("e7", mock)
    names = {"cars": "Ontology", "cars_flat_kb": "Flat KB", "cars_oracle_kb": "Oracle (reference)"}
    if not df.empty:
        tab = []
        for p, d in df.groupby("policy"):
            runs = d.groupby("run_key").real_compromise.sum()
            forms = [dec for ds in d.decisions for (_, dec) in ds]
            lab = d.atlas_correct.dropna()
            tab.append({"Knowledge layer": names.get(p, p),
                        "Real compromises / run": fmt_ci(*mean_ci(list(runs)), digits=1),
                        "ATLAS label accuracy %": f"{100 * lab.mean():.1f}" if len(lab) else "--",
                        "Pending forms": sum(f.endswith("pending") for f in forms),
                        "Success forms": sum(f.endswith("success") for f in forms),
                        "Exposures": int(d.exposed.sum()), "Runs": len(runs)})
        write_table(pd.DataFrame(tab), f"table_live_e7{sfx}",
                    "Live E7: CARS with ontology vs. an otherwise identical flat knowledge layer.", f"tab:live-e7{sfx}")
    # decision audit: which decisions does the ontology change, and do those changes matter?
    f = LIVE / ("e7_decision_diff_mock.json" if mock else "e7_decision_diff.json")
    if not f.exists():
        return
    au = json.loads(f.read_text())
    rows = au["rows"]
    ch = [r for r in rows if r["form_changed"] or r["atlas_changed"]]
    tab = [{"Tool": r["tool"] + (" (honeytool)" if r["honeytool"] else ""),
            "Verifiability flat / onto / oracle": f"{r['ver_flat']:.2f} / {r['ver_onto']:.2f} / {r['ver_oracle']:.2f}",
            "ATLAS flat -> onto (oracle)": f"{r['atlas_flat']} -> {r['atlas_onto']} ({r['atlas_oracle']})",
            "Form changed": "yes" if r["form_changed"] else "no",
            "Label now correct": ("yes" if r["atlas_correct_onto"] and not r["atlas_correct_flat"] else
                                  "no, now wrong" if r["atlas_correct_flat"] and not r["atlas_correct_onto"] else "unchanged")}
           for r in ch]
    summ = {"Tool": f"SUMMARY ({len(rows)} tools)", "Verifiability flat / onto / oracle":
            f"mean abs. error flat {np.mean([r['ver_abs_err_flat'] for r in rows]):.3f}, "
            f"onto {np.mean([r['ver_abs_err_onto'] for r in rows]):.3f}",
            "ATLAS flat -> onto (oracle)": f"correct flat {sum(r['atlas_correct_flat'] for r in rows)}, "
                                           f"onto {sum(r['atlas_correct_onto'] for r in rows)}",
            "Form changed": f"{sum(r['form_changed'] for r in rows)} tools", "Label now correct": ""}
    write_table(pd.DataFrame(tab + [summ]), f"table_live_e7_decisions{sfx}",
                f"E7 decision audit on {au['suite']}: tools whose CARS decision (response form over v_hat in "
                f"{au['vhat_grid']}) or alert label differs between flat KB and ontology. Tools with no change are "
                "summarised only.", f"tab:live-e7-dec{sfx}")
    # paired outcome effect on the rounds where the two layers actually decided differently
    if df.empty or not {"cars", "cars_flat_kb"} <= set(df.policy):
        return
    k = ["seed", "round"]
    a_ = df[df.policy == "cars"].set_index(k)
    b_ = df[df.policy == "cars_flat_kb"].set_index(k)
    j = a_[["decisions", "real_compromise", "exposed"]].add_suffix("_onto").join(
        b_[["decisions", "real_compromise", "exposed"]].add_suffix("_flat"), how="inner")
    j["differs"] = j.decisions_onto.astype(str) != j.decisions_flat.astype(str)
    out = []
    for lab, x in (("rounds with different decisions", j[j.differs]), ("rounds with identical decisions", j[~j.differs])):
        out.append({"Paired rounds": lab, "n": len(x),
                    "Real compromise onto - flat": f"{(x.real_compromise_onto.astype(int) - x.real_compromise_flat.astype(int)).sum():+d}" if len(x) else "--",
                    "Exposures onto - flat": f"{(x.exposed_onto.astype(int) - x.exposed_flat.astype(int)).sum():+d}" if len(x) else "--"})
    write_table(pd.DataFrame(out), f"table_live_e7_paired{sfx}",
                "E7 paired rounds (same seed and round, common random numbers): outcome differences where the "
                "ontology changed a decision vs. where it did not. Note decisions can also diverge downstream of "
                "an earlier different decision.", f"tab:live-e7-paired{sfx}")


def e5(sfx: str, mock: bool, e3df: pd.DataFrame):
    df = _load("e5", mock)
    if df.empty:
        print("  (no E5 data)")
        return
    if "condition" not in df:
        df["condition"] = "clean"
    df["condition"] = df.condition.fillna("clean")
    tab = []
    for su in sorted(df.suite.unique()):
        for p in _order(df.policy.unique()):
            cl = df[(df.suite == su) & (df.policy == p) & (df.condition == "clean")]
            co = df[(df.suite == su) & (df.policy == p) & (df.condition == "contaminated")]
            calls = co.n_calls.sum() if len(co) else 0
            tab.append({
                "Suite": su, "Policy": POLICY_LABELS.get(p, p),
                "Clean: fabrications in trusted flows (must be 0)": int(cl.n_fabricated.sum()),
                "Clean: benign utility %": f"{100 * cl.utility.mean():.1f}" if len(cl) else "--",
                "Clean: latency s": f"{cl.latency_s.mean():.2f}" if len(cl) else "--",
                "Contaminated: benign calls misrouted %": f"{100 * co.benign_misrouted.sum() / max(1, calls):.2f}" if len(co) else "--",
                "Contaminated: benign calls given fabricated results": int(co.benign_fabricated.sum()) if len(co) else "--",
                "Contaminated: utility %": f"{100 * co.utility.mean():.1f}" if len(co) else "--",
                "Contaminated: real compromise %": f"{100 * co.real_compromise.mean():.1f}" if len(co) else "--",
                "Episodes clean / contam.": f"{len(cl)} / {len(co)}"})
    write_table(pd.DataFrame(tab), f"table_live_e5{sfx}",
                "E5 (real agent): benign safety in two separate conditions. Clean = authenticated legitimate tasks "
                "with no untrusted instructions; tests the R1 invariant and utility. Contaminated = the same tasks "
                "with untrusted injected data present; tests false routing of the user's own benign actions.",
                f"tab:live-e5{sfx}")
    v = df[(df.condition == "clean") & ((df.n_fabricated > 0) | (df.n_detected > 0))]
    if not v.empty:
        write_table(pd.DataFrame([{"Suite": r.suite, "Policy": POLICY_LABELS.get(r.policy, r.policy),
                                   "User task": r.user_task, "Flagged calls": r.n_detected,
                                   "Fabricated": r.n_fabricated, "Utility": r.utility,
                                   "Decisions": "; ".join(f"{f}:{d}" for f, d in r.decisions)}
                                  for r in v.itertuples()]), f"table_live_e5_flags{sfx}",
                    "E5 clean episodes in which the gate flagged a legitimate call (false positives). Rows with "
                    "Fabricated > 0 violate the R1 invariant and must be reported.", f"tab:live-e5-flags{sfx}")


def e6(sfx: str, mock: bool, e3df: pd.DataFrame, cfg: dict):
    if e3df.empty:
        return
    mins = cfg["e6"]["triage_minutes_per_alert"]
    tab = []
    for p in _order(e3df.policy.unique()):
        d = e3df[e3df.policy == p]
        lab = d.atlas_correct.dropna()
        alerts = int((d.n_detected > 0).sum())
        tab.append({"Policy": POLICY_LABELS.get(p, p), "Episodes": len(d), "Alerts": alerts,
                    "ATLAS label accuracy %": f"{100 * lab.mean():.1f}" if len(lab) else "--",
                    "Intelligence I (deception interactions)": int(d.n_fabricated.sum() + d.honeytool_calls.sum()),
                    "Analyst minutes": alerts * mins,
                    "Real compromises": int(d.real_compromise.sum())})
    write_table(pd.DataFrame(tab), f"table_live_e6{sfx}",
                f"E6: operational value. Triage assumes {mins} analyst minutes per alert. ATLAS techniques: "
                + "; ".join(f"{k} {v}" for k, v in ATLAS_NAMES.items()) + ".", f"tab:live-e6{sfx}")


def spend(mock: bool):
    p = LIVE / ("usage_mock.json" if mock else "usage.json")
    if not p.exists():
        return
    u = json.loads(p.read_text())
    rows = [{"Model": m, "Calls": v["calls"], "Input tokens": v["input"], "Output tokens": v["output"],
             "Cost USD": f"{v['usd']:.2f}"} for m, v in u["by_model"].items()]
    rows.append({"Model": "TOTAL", "Calls": sum(v["calls"] for v in u["by_model"].values()),
                 "Input tokens": sum(v["input"] for v in u["by_model"].values()),
                 "Output tokens": sum(v["output"] for v in u["by_model"].values()), "Cost USD": f"{u['total_usd']:.2f}"})
    write_table(pd.DataFrame(rows), "table_live_spend" + ("_mock" if mock else ""),
                "Measured API usage and cost of the live experiments.", "tab:live-spend")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    global PROFILE
    PROFILE = "_permissive" if cfg["agent"].get("profile") == "permissive" else ""
    sfx = "_mock" if a.mock else PROFILE
    print("=== live report ===")
    e0(sfx, a.mock)
    e3df = e3(sfx, a.mock)
    e3v(sfx, a.mock)
    e2(sfx, a.mock)
    e4(sfx, a.mock)
    e7(sfx, a.mock)
    e5(sfx, a.mock, e3df)
    e6(sfx, a.mock, e3df, cfg)
    spend(a.mock)


if __name__ == "__main__":
    main()
