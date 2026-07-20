""""
compas_grasp_bic.py
===================
Quantifies how strongly the BIC prefers GRaSP's temporally impossible
COMPAS structure over the corrected orientation (Section VII of the paper).

Procedure:
  1. Run GRaSP (pinned initialization) on the nine-variable COMPAS frame.
  2. Extract its directed edges; orient any undirected CPDAG edges by a fixed
     deterministic rule (identical in both graphs, so they cancel in the delta).
  3. Build the "corrected" graph: same skeleton, with every edge directed INTO
     an immutable attribute (Race, Sex, Age) reversed.
  4. Score both DAGs with the decomposable Gaussian local BIC
     (n*ln(RSS/n) + k*ln(n) per node; lower = better) and report the delta.

Delta = BIC(corrected) - BIC(GRaSP). Positive => the BIC prefers GRaSP's
impossible structure, i.e., the search sits in a local optimum that edge
reversals cannot escape.

Usage:
    python compas_grasp_bic.py            # rebuilds the frame via compas_analysis
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

IMMUTABLE = {"Race", "Sex", "Age"}
GRASP_SEED = 42

# Deterministic orientation for undirected CPDAG edges: later in this order
# receives the arrow (temporal plausibility: demographics -> record -> score -> recid).
ORIENT_ORDER = ["Race", "Sex", "Age", "JuvFel", "JuvMisd", "Priors",
                "ChargeDeg", "Score", "Recid"]


def load_frame() -> pd.DataFrame:
    """Build the nine-variable numeric analysis frame (n=5,278) directly from
    the raw ProPublica CSV, applying the paper's documented preprocessing.
    Self-contained on purpose: this verification script should not depend on
    the project loader's encoding stage."""
    import os
    candidates = ["data/compas-scores-two-years.csv", "compas-scores-two-years.csv"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            "compas-scores-two-years.csv not found; run main_compas.py once "
            "to download it, or place it under data/.")
    df = pd.read_csv(path)

    # Standard ProPublica filters (Section VI-A of the paper) -> n = 6,172
    df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30)]
    df = df[df.is_recid != -1]
    df = df[df.c_charge_degree.isin(["F", "M"])]
    df = df[df.score_text != "N/A"]
    # Restrict to African-American and Caucasian defendants -> n = 5,278
    df = df[df.race.isin(["African-American", "Caucasian"])]

    frame = pd.DataFrame({
        "Race": (df.race == "African-American").astype(float),
        "Sex": (df.sex == "Male").astype(float),
        "Age": df.age.astype(float),
        "JuvFel": df.juv_fel_count.astype(float),
        "JuvMisd": df.juv_misd_count.astype(float),
        "Priors": df.priors_count.astype(float),
        "ChargeDeg": (df.c_charge_degree == "F").astype(float),
        "Score": df.decile_score.astype(float),
        "Recid": df.two_year_recid.astype(float),
    }).reset_index(drop=True)

    # Guards: catch frame-construction drift before it reaches GRaSP.
    assert len(frame) == 5278, f"expected n=5278, got {len(frame)}"
    bad = [c for c in frame.columns if frame[c].dtype.kind not in "fi"]
    assert not bad, f"non-numeric columns would break np.cov: {bad}"
    print(f"Frame OK: n={len(frame)} | AA mean score="
          f"{frame[frame.Race == 1].Score.mean():.2f} | "
          f"Cau mean score={frame[frame.Race == 0].Score.mean():.2f}")
    return frame


def grasp_edges(frame: pd.DataFrame):
    """Run pinned GRaSP; return (directed, undirected) edge lists."""
    from causallearn.search.PermutationBased.GRaSP import grasp

    cols = list(frame.columns)
    random.seed(GRASP_SEED)
    G = grasp(frame.values, score_func="local_score_BIC")
    A = G.graph
    directed, undirected = [], []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if A[j, i] == 1 and A[i, j] == -1:
                directed.append((cols[i], cols[j]))
            elif A[i, j] == 1 and A[j, i] == -1:
                directed.append((cols[j], cols[i]))
            elif A[i, j] != 0 or A[j, i] != 0:
                undirected.append((cols[i], cols[j]))
    return directed, undirected


def build_parent_maps(directed, undirected, cols):
    """Parent maps for GRaSP's graph and the immutable-corrected graph."""
    rank = {v: k for k, v in enumerate(ORIENT_ORDER)}
    shared = [(a, b) if rank[a] < rank[b] else (b, a) for a, b in undirected]

    grasp_p = {c: [] for c in cols}
    for s, t in directed + shared:
        grasp_p[t].append(s)

    corrected = [(t, s) if t in IMMUTABLE else (s, t) for s, t in directed]
    corr_p = {c: [] for c in cols}
    for s, t in corrected + shared:
        corr_p[t].append(s)

    _assert_acyclic(corr_p, cols)
    return grasp_p, corr_p


def _assert_acyclic(parents, cols):
    seen, done = set(), set()

    def visit(v):
        if v in done:
            return
        if v in seen:
            raise ValueError(f"Reversal created a cycle at {v}; "
                             "choose a different correction rule.")
        seen.add(v)
        for p in parents[v]:
            visit(p)
        done.add(v)

    for c in cols:
        visit(c)


def local_bic(frame, child, parents):
    """Gaussian local BIC: n*ln(RSS/n) + k*ln(n). Lower = better."""
    n = len(frame)
    y = frame[child].values
    X = np.column_stack([np.ones(n)] + [frame[p].values for p in parents]) \
        if parents else np.ones((n, 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((y - X @ beta) ** 2).sum())
    return n * np.log(rss / n) + len(parents) * np.log(n)


def main():
    frame = load_frame()
    cols = list(frame.columns)
    directed, undirected = grasp_edges(frame)

    print("GRaSP directed edges (pinned seed %d):" % GRASP_SEED)
    for s, t in directed:
        flag = "  <-- INTO immutable" if t in IMMUTABLE else ""
        print(f"  {s} -> {t}{flag}")
    print("Undirected (oriented identically in both graphs):", undirected)

    grasp_p, corr_p = build_parent_maps(directed, undirected, cols)
    n_edges = sum(len(v) for v in grasp_p.values())
    assert n_edges == sum(len(v) for v in corr_p.values()), "edge counts differ"

    bic_g = sum(local_bic(frame, c, p) for c, p in grasp_p.items())
    bic_c = sum(local_bic(frame, c, p) for c, p in corr_p.items())
    print(f"\nEdges in each graph: {n_edges}")
    print(f"BIC (GRaSP structure)    : {bic_g:.1f}")
    print(f"BIC (corrected structure): {bic_c:.1f}")
    print(f"DeltaBIC = corrected - GRaSP: {bic_c - bic_g:+.1f}")
    print("Positive => BIC prefers GRaSP's temporally impossible structure;")
    print("the search sits in a local optimum that edge reversals cannot escape.")

    print("\nPer-node contributions (corrected - GRaSP):")
    for c in cols:
        if set(grasp_p[c]) != set(corr_p[c]):
            d = local_bic(frame, c, corr_p[c]) - local_bic(frame, c, grasp_p[c])
            print(f"  {c:10s} {d:+8.1f}")


if __name__ == "__main__":
    main()