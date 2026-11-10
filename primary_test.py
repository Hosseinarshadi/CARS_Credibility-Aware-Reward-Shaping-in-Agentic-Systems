"""Pre-specified primary hypothesis test for the paper (confirmatory analysis).

Primary family (fixed before looking at the data, 4 hypotheses):
    H_v : at full assumed compliance (c = 1.0), full CARS produces the same number of real
          compromises per run as budget-matched fixed shaping, for v in {0.1, 0.3, 0.5, 0.8}.
Test:      two-sided exact sign test on paired run-level differences (CARS - matched),
           pairs = same suite, attacker algorithm and seed (common random numbers).
Correction: Holm-Bonferroni over the 4 hypotheses (controls the family-wise error rate at 5%).
Everything else in the paper is exploratory (reported with CIs, unadjusted).

    python -m experiments.primary_test           # from results/live/e3v.jsonl
    python -m experiments.primary_test --mock    # from the mock logs
Writes tables/table_primary_test.csv / .tex
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import pandas as pd

from .common import mean_ci, write_table

LIVE = Path(__file__).resolve().parents[1] / "results" / "live"


def sign_test(d):
    neg, pos = sum(x < 0 for x in d), sum(x > 0 for x in d)
    n = neg + pos
    if n == 0:
        return 1.0, neg, pos
    k = min(neg, pos)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), neg, pos


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--compliance", type=float, default=1.0)
    ap.add_argument("--ref", default="cars")
    ap.add_argument("--other", default="fixed_matched")
    a = ap.parse_args()
    f = LIVE / ("e3v_mock.jsonl" if a.mock else "e3v.jsonl")
    df = pd.DataFrame([json.loads(l) for l in f.open()])
    runs = df.groupby("run_key").agg(suite=("suite", "first"), policy=("policy", "first"),
                                     algo=("attacker_algo", "first"), seed=("seed", "first"),
                                     c=("compliance_set", "first"), v=("v_att", "first"),
                                     C=("real_compromise", "sum"), nfab=("n_fabricated", "sum")).reset_index()
    runs = runs[runs.c.astype(float) == a.compliance]
    rows, ps = [], []
    for v in sorted(runs.v.unique()):
        sub = runs[runs.v == v]
        k = ["suite", "algo", "seed"]
        x = sub[sub.policy == a.ref].set_index(k)
        y = sub[sub.policy == a.other].set_index(k)
        j = x[["C", "nfab"]].join(y[["C", "nfab"]], lsuffix="_ref", rsuffix="_oth", how="inner")
        d = list(j.C_ref - j.C_oth)
        m, lo, hi = mean_ci(d)
        p, neg, pos = sign_test(d)
        ps.append(p)
        rows.append({"v": v, "Pairs": len(d), "Mean diff (CARS - matched)": f"{m:.2f}",
                     "95% CI": f"[{lo:.2f}, {hi:.2f}]", "CARS fewer / more / tied": f"{neg} / {pos} / {len(d) - neg - pos}",
                     "Sign test p": p,
                     "Fabrications/run CARS vs matched": f"{j.nfab_ref.mean():.2f} vs {j.nfab_oth.mean():.2f}"})
    for r, ph in zip(rows, holm(ps)):
        r["Holm-adjusted p"] = ph
        r["Reject H0 at 0.05"] = "yes" if ph < 0.05 else "no"
        r["Sign test p"] = f"{r['Sign test p']:.4f}"
        r["Holm-adjusted p"] = f"{ph:.4f}"
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    write_table(out, "table_primary_test" + ("_mock" if a.mock else ""),
                f"Primary confirmatory test (pre-specified family of 4): CARS vs budget-matched fixed shaping at "
                f"c = {a.compliance:g}, paired by suite, attacker algorithm and seed. Exact two-sided sign test, "
                "Holm-Bonferroni adjusted over the family. Bootstrap 95% CI for the mean paired difference.",
                "tab:primary")


if __name__ == "__main__":
    main()
