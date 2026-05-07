"""
main_compas.py
==============
Reproduce Study 2 (Section VI): COMPAS recidivism analysis.

Pipeline:
  1. Load raw ProPublica COMPAS CSV (downloads on first run, then caches).
  2. Apply ProPublica's standard preprocessing filters.
  3. Compute baseline correlation-based fairness metrics (DIR, parity).
  4. Run all six causal discovery algorithms and tabulate Race -> Score
     and Race -> Priors detections (Table III reproduction).
  5. Compute shared node positions ONCE from all results (so every
     individual figure and the grid use identical node coordinates).
  6. Plot individual DAGs and the algorithm grid.
  7. Compute backdoor-adjusted ATE with progressive control sets.

KEY CHANGE from previous version
---------------------------------
Both individual plots AND the grid now share the SAME node positions,
computed once by ``compute_shared_pos(results, layout="kamada")``.
This guarantees that every figure — whether a single-algorithm panel
or the 6-panel grid — places nodes identically, making visual
comparison trivial for readers and reviewers.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from compas_analysis import (
    load_compas,
    preprocess_compas,
    baseline_disparities,
    plot_correlation_vs_causal,
)
from causal_discovery import run_all, report_convention
from ate_estimation import (
    backdoor_ate,
    staged_backdoor_ate,
    disparate_impact_ratio,
    statistical_parity,
)
from visualization import (
    plot_discovery_result,
    plot_grid,
    compute_shared_pos,        # ← new shared-position helper
    DEFAULT_ROLES_COMPAS,
)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("=" * 72)
    print("STUDY 2: COMPAS RECIDIVISM ANALYSIS")
    print("=" * 72)

    # Detect causal-learn's edge-encoding convention before any algorithm
    # runs. The auto-detector runs PC on a known X->Y->Z chain and picks
    # the matrix interpretation consistent with the result, so the plotted
    # arrows always match the algorithms' actual outputs regardless of
    # which causal-learn version (or convention) is installed.
    print()
    report_convention()

    # 1. Load + preprocess
    raw = load_compas()
    df  = preprocess_compas(raw)
    print(f"\nPreprocessed COMPAS shape: {df.shape}")
    print(df.describe().round(3))

    # 2. Baseline disparities
    print("\n=== Baseline disparities (correlation-based) ===")
    base = baseline_disparities(df)
    for k, v in base.items():
        print(f"  {k:25s} {v}")

    # 3. Causal discovery
    print("\n=== Running causal discovery on COMPAS ===")
    results = run_all(
        df,
        direct_lingam_exogenous=["Race", "Sex", "Age"],
        direct_lingam_sinks=["Recidivism"],
    )
    for name, res in results.items():
        if res is None:
            continue
        print(res.summary())
        print()

    # 4. Tabulate Race->Score and Race->Priors detections (Table IV)
    rows = []
    for name, res in results.items():
        if res is None:
            rows.append({
                "algorithm":    name,
                "Race->Score":  "ERROR",
                "Race->Priors": "ERROR",
                "lat_conf":     False,
                "beta_hat":     None,
            })
            continue
        rows.append({
            "algorithm":    name,
            "Race->Score":  "YES" if res.has_directed_edge("Race", "Score") else "no",
            "Race->Priors": "YES" if res.has_directed_edge("Race", "Priors") else "no",
            "lat_conf":     bool(res.bidirected_edges),
            "beta_hat":     (round(res.get_coefficient("Race", "Score"), 4)
                             if res.coef_matrix is not None else None),
        })
    table4 = pd.DataFrame(rows)
    print("\n=== Table IV: COMPAS algorithm comparison ===")
    print(table4.to_string(index=False))
    table4.to_csv("results/compas_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # 5. Compute shared node positions ONCE
    # -------------------------------------------------------------------------
    # compute_shared_pos() prefers the domain-aware fixed layout (outcomes on
    # the far right at x=10, protected attributes on the far left at x=0).
    # This enforces a clear left→right causal direction in every figure.
    # Falls back to kamada-kawai only if the variable set is unrecognised.
    print("\n--- Computing shared node positions (fixed layout: outcomes right) ---")
    shared_pos = compute_shared_pos(results)   # layout="fixed" by default
    print(f"  Positions computed for: {list(shared_pos.keys())}")

    # -------------------------------------------------------------------------
    # 6. DAG figures — individual + grid
    # -------------------------------------------------------------------------
    flagged = [("Race", "Score"), ("Race", "Priors")]

    print("\n--- Plotting individual DAGs ---")
    for name, res in results.items():
        if res is None:
            continue
        plot_discovery_result(
            res,
            title       = f"{name} on COMPAS (n={len(df)})",
            flagged_edges = flagged,
            node_roles  = DEFAULT_ROLES_COMPAS,
            pos         = shared_pos,       # ← same positions as the grid
            save_path   = f"figures/compas_{name.replace('-', '')}.pdf",
        )
        plt.close()

    print("\n--- Plotting algorithm grid ---")
    plot_grid(
        results,
        flagged_edges  = flagged,
        node_roles     = DEFAULT_ROLES_COMPAS,
        title          = "All algorithms on COMPAS",
        pos            = shared_pos,        # ← same positions as individuals
        save_path      = "figures/compas_grid.pdf",
    )
    plt.close("all")

    # -------------------------------------------------------------------------
    # 7. Backdoor-adjustment ATE (Equations 1-3 in the paper)
    # -------------------------------------------------------------------------
    print("\n=== Backdoor ATE: Race -> Score ===")
    print("Formula: ATE = E_Z[ E[Y|T=1,Z] - E[Y|T=0,Z] ]")
    print("Under linearity: ATE_hat = OLS coefficient on Race after controlling for Z")

    stages = [
        ("no controls",           []),
        ("age + charge",          ["Age", "ChargeDegree"]),
        ("full criminal history", ["Age", "ChargeDegree",
                                   "JuvFelony", "JuvMisd", "Priors"]),
    ]
    ate_table = staged_backdoor_ate(df, "Race", "Score", stages)
    print(ate_table.to_string(index=False))
    ate_table.to_csv("results/compas_ate.csv", index=False)

    # -------------------------------------------------------------------------
    # 8. Side-by-side: correlation-based vs causal estimates
    # -------------------------------------------------------------------------
    print("\n=== Correlation-based vs causal estimates for Race -> Score ===")
    dir_score = disparate_impact_ratio(df, "Race", "Score")
    sp_score  = statistical_parity(df, "Race", "Score")
    ate_full, _ = backdoor_ate(
        df, "Race", "Score",
        ["Age", "ChargeDegree", "JuvFelony", "JuvMisd", "Priors"],
    )
    print(f"  DIR (Score)                = {dir_score:.3f}")
    print(f"  Statistical parity (Score) = {sp_score:+.3f}")
    print(f"  Raw ATE (no controls)      = {ate_table.iloc[0]['ATE']:+.3f}")
    print(f"  ATE (full controls)        = {ate_full:+.3f}")

    beta_dl = beta_ica = None
    if results.get("DirectLiNGAM") is not None:
        beta_dl = results["DirectLiNGAM"].get_coefficient("Race", "Score")
        print(f"  DirectLiNGAM beta_hat      = {beta_dl:+.4f}")
    if results.get("ICA-LiNGAM") is not None:
        beta_ica = results["ICA-LiNGAM"].get_coefficient("Race", "Score")
        print(f"  ICA-LiNGAM   beta_hat      = {beta_ica:+.4f}")

    # Correlation vs causal comparison figure
    print("\n=== Generating correlation-vs-causal comparison figure ===")
    correlation_metrics = {
        "COMPAS Score\nDisparate Impact\n(\u22650.8 = fair)": base["DIR_score"],
        "COMPAS Score\nStat. Parity\n(=0 = fair)":           base["stat_parity_score"],
        "Recidivism\nDisparate Impact":                       base["DIR_recidivism"],
    }
    causal_estimates = {
        "ATE Race\u2192Score\n(no controls)":   ate_table.iloc[0]["ATE"],
        "ATE Race\u2192Score\n(full controls)": ate_full,
    }
    if beta_dl is not None:
        causal_estimates["DirectLiNGAM \u03b2\nRace\u2192Score"] = beta_dl
    if beta_ica is not None:
        causal_estimates["ICA-LiNGAM \u03b2\nRace\u2192Score"]   = beta_ica

    plot_correlation_vs_causal(
        correlation_metrics = correlation_metrics,
        causal_estimates    = causal_estimates,
        save_path           = "figures/compas_corr_vs_causal",
    )

    print("\nAll outputs written to results/ and figures/")


if __name__ == "__main__":
    main()
