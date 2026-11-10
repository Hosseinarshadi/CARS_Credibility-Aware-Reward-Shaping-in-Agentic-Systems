"""Score expert annotations of tool verifiability / ATLAS technique and re-evaluate the
ontology against the expert consensus (replaces the provisional author labels).

1. Each security expert fills a copy of cars/ontology/annotation_template.csv
   (annotator_id, verifiability_0_to_1, atlas_technique) and saves it in annotations/.
2. python -m experiments.annotation_agreement annotations/*.csv
Reports Fleiss' kappa on binned verifiability (low < 0.4 <= mid < 0.7 <= high) and on the
ATLAS technique, then the ontology and flat-KB errors against the consensus.
"""
import sys
from collections import Counter, defaultdict

import pandas as pd

from cars.ontology.reasoner import infer


def fleiss(table):
    """table: list of Counter(category -> count), same number of raters per item."""
    n = sum(table[0].values())
    cats = sorted({c for t in table for c in t})
    N = len(table)
    p = [sum(t[c] for t in table) / (N * n) for c in cats]
    P = [(sum(v * v for v in t.values()) - n) / (n * (n - 1)) for t in table]
    Pbar, Pe = sum(P) / N, sum(x * x for x in p)
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else 1.0


def binv(v):
    return "low" if v < 0.4 else ("mid" if v < 0.7 else "high")


def main(paths):
    df = pd.concat([pd.read_csv(p) for p in paths]).dropna(subset=["verifiability_0_to_1"])
    items = defaultdict(list)
    for _, r in df.iterrows():
        items[r.tool].append(r)
    k = Counter(len(v) for v in items.values()).most_common(1)[0][0]
    full = {t: v for t, v in items.items() if len(v) == k}
    print(f"{len(full)} tools rated by {k} annotators")
    if k >= 2:
        print("Fleiss kappa, verifiability bins:", round(fleiss([Counter(binv(float(r.verifiability_0_to_1)) for r in v) for v in full.values()]), 3))
        print("Fleiss kappa, ATLAS technique:   ", round(fleiss([Counter(str(r.atlas_technique) for r in v) for v in full.values()]), 3))
    rows = []
    for t, v in full.items():
        cons_v = sum(float(r.verifiability_0_to_1) for r in v) / len(v)
        cons_a = Counter(str(r.atlas_technique) for r in v).most_common(1)[0][0]
        o = infer(t.split(":")[-1], use_assertions=False)
        rows.append({"tool": t, "consensus_v": cons_v, "onto_err": abs(o["verifiability"] - cons_v),
                     "flat_err": abs(0.5 - cons_v), "onto_atlas_ok": o["atlas"] == cons_a,
                     "flat_atlas_ok": cons_a == "AML.T0053"})
    r = pd.DataFrame(rows)
    print(f"Ontology MAE {r.onto_err.mean():.3f} vs flat {r.flat_err.mean():.3f}; "
          f"ATLAS accuracy ontology {100 * r.onto_atlas_ok.mean():.0f}% vs flat {100 * r.flat_atlas_ok.mean():.0f}%")
    r.to_csv("tables/table_e7_expert_rescore.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
