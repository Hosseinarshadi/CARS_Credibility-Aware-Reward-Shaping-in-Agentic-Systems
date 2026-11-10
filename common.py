"""Shared helpers: parallel execution, result storage, summary statistics, plotting style."""
from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
for d in (RESULTS, FIGURES, TABLES):
    d.mkdir(exist_ok=True)

POLICY_LABELS = {
    "block": "Blocking only",
    "static_honeytool": "Static honeytool",
    "fixed_shaping": "Fixed reward shaping",
    "fixed_matched": "Fixed shaping, CARS-matched budget",
    "state_consistent_adaptive": "State-consistent adaptive*",
    "cars_noverif": "CARS w/o verification",
    "cars": "CARS (full)",
    "cars_minus_stat": "CARS - statistical",
    "cars_minus_sem": "CARS - semantic",
    "cars_minus_ver": "CARS - verifiability",
    "cars_minus_pop": "CARS - population",
}
POLICY_COLORS = {
    "block": "#7f7f7f", "static_honeytool": "#bcbd22", "fixed_shaping": "#ff7f0e", "fixed_matched": "#c49c00",
    "state_consistent_adaptive": "#9467bd", "cars_noverif": "#17becf", "cars": "#1f4e9c",
    "cars_minus_stat": "#d62728", "cars_minus_sem": "#e377c2", "cars_minus_ver": "#8c564b",
    "cars_minus_pop": "#2ca02c",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def workers() -> int:
    return int(os.environ.get("CARS_WORKERS", os.cpu_count() or 1))


def run_many(fn, configs: list, desc: str = "") -> list[dict]:
    """Run fn(config) for every config in parallel, preserving order."""
    n = workers()
    print(f"[{desc}] {len(configs)} runs on {n} workers", flush=True)
    if n <= 1:
        return [fn(c) for c in configs]
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, configs, chunksize=max(1, len(configs) // (n * 8))))


def save_jsonl(rows: list[dict], name: str) -> Path:
    p = RESULTS / f"{name}.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return p


def load_jsonl(name: str) -> list[dict]:
    p = RESULTS / f"{name}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def mean_ci(xs, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean and bootstrap 95% CI. Returns (mean, lo, hi)."""
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return (float("nan"),) * 3
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, m, m
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    return m, boots[int(alpha / 2 * n_boot)], boots[int((1 - alpha / 2) * n_boot) - 1]


def diff_ci(a, b, n_boot: int = 2000, seed: int = 0):
    """Bootstrap CI of mean(a) - mean(b) for independent samples."""
    rng = random.Random(seed)
    d = sorted(sum(rng.choice(a) for _ in a) / len(a) - sum(rng.choice(b) for _ in b) / len(b)
               for _ in range(n_boot))
    return sum(a) / len(a) - sum(b) / len(b), d[int(0.025 * n_boot)], d[int(0.975 * n_boot) - 1]


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (well behaved near 0 and 1)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt_ci(m, lo, hi, digits: int = 0) -> str:
    if math.isnan(m):
        return "--"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_table(df: pd.DataFrame, name: str, caption: str, label: str):
    """Write CSV + booktabs LaTeX (LNCS-ready)."""
    df.to_csv(TABLES / f"{name}.csv", index=False)
    cols = "l" * len(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\scriptsize", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(_tex(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_tex(str(v)) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"  table -> tables/{name}.csv / .tex")


def _tex(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
             .replace("&", r"\&").replace("#", r"\#"))


def savefig(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    print(f"  figure -> figures/{name}.pdf / .png")


def settings(quick: bool) -> dict:
    """Run sizes. quick=True is a smoke test (minutes); default is paper scale."""
    if quick:
        return {"runs": 4, "budget": 4000, "budget_small": 3000}
    return {"runs": int(os.environ.get("CARS_RUNS", 30)), "budget": 20000, "budget_small": 8000}
