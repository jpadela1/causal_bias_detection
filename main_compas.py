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
  5. Plot DAGs and the LiNGAM-annotated graph.
  6. Compute backdoor-adjusted ATE with progressive control sets:
        ATE_no_controls   (raw correlation)
        ATE_age + charge  (partial controls)
        ATE_full_history  (full controls)
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
from causal_discovery import run_all
from ate_estimation import (
    backdoor_ate,
    staged_backdoor_ate,
    disparate_impact_ratio,
    statistical_parity,
)
from visualization import plot_discovery_result, plot_grid, DEFAULT_ROLES_COMPAS


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("=" * 72)
    print("STUDY 2: COMPAS RECIDIVISM ANALYSIS")
    print("=" * 72)

    # 1. Load + preprocess
    raw = load_compas()
    df = preprocess_compas(raw)
    print(f"\nPreprocessed COMPAS shape: {df.shape}")
    print(df.describe().round(3))

    # 2. Baseline disparities
    print("\n=== Baseline disparities (correlation-based) ===")
    base = baseline_disparities(df)
    for k, v in base.items():
        print(f"  {k:25s} {v}")

    # 3. Causal discovery
    print("\n=== Running causal discovery on COMPAS ===")
    # COMPAS has discrete variables (binary charge degree, integer priors,
    # binary recidivism), which violates LiNGAM's continuous-noise assumption.
    # Without prior knowledge, DirectLiNGAM produces nonsensical orderings
    # like ChargeDegree -> Race. We anchor immutable demographics as
    # exogenous variables and Recidivism as the ultimate sink.
    results = run_all(
        df,
        direct_lingam_exogenous=["Race", "Sex", "Age"],
        direct_lingam_sinks=["Recidivism","Score"],
    )
    for name, res in results.items():
        if res is None:
            continue
        print(res.summary())
        print()

    # Tabulate Race->Score and Race->Priors detections (Table III)
    rows = []
    for name, res in results.items():
        if res is None:
            rows.append({"algorithm": name, "Race->Score": "ERROR",
                         "Race->Priors": "ERROR", "lat_conf": False, "beta_hat": None})
            continue
        rows.append({
            "algorithm": name,
            "Race->Score": "YES" if res.has_directed_edge("Race", "Score") else "no",
            "Race->Priors": "YES" if res.has_directed_edge("Race", "Priors") else "no",
            "lat_conf": bool(res.bidirected_edges),
            "beta_hat": (round(res.get_coefficient("Race", "Score"), 4)
                         if res.coef_matrix is not None else None),
        })
    table3 = pd.DataFrame(rows)
    print("\n=== Table III: COMPAS algorithm comparison ===")
    print(table3.to_string(index=False))
    table3.to_csv("results/compas_summary.csv", index=False)

#end Table III block and adding new SHD computation code
    from compas_analysis import get_compas_ground_truth
    from causal_discovery import structural_hamming_distance

    gt_biased = get_compas_ground_truth("biased")
    gt_fair = get_compas_ground_truth("fair")
    for name, res in results.items():
        if res is None:
            continue
        shd_b = structural_hamming_distance(res, gt_biased)
        shd_f = structural_hamming_distance(res, gt_fair)
        print(f"  {name:14s}  SHD-biased={shd_b}  SHD-fair={shd_f}")

    # End ddding SHD computation


    # 4. DAG figures
    flagged = [("Race", "Score"), ("Race", "Priors")]
    for name, res in results.items():
        if res is None:
            continue
        plot_discovery_result(
            res,
            title=f"{name} on COMPAS (n={len(df)})",
            flagged_edges=flagged,
            node_roles=DEFAULT_ROLES_COMPAS,
            save_path=f"figures/compas_{name.replace('-', '')}.png",
            layout="kamada",
        )
        plt.close()
    plot_grid(
        results,
        flagged_edges=flagged,
        node_roles=DEFAULT_ROLES_COMPAS,
        title="All algorithms on COMPAS",
        save_path="figures/compas_grid.png",
        layout="kamada",
    )
    plt.close("all")

    # 5. Backdoor-adjustment ATE (Equations 1-3 in the paper)
    print("\n=== Backdoor ATE: Race -> Score ===")
    print("Formula: ATE = E_Z[ E[Y|T=1,Z] - E[Y|T=0,Z] ]")
    print("Under linearity:  ATE_hat = OLS coefficient on T (=Race) after controlling for Z")

    stages = [
        ("no controls", []),
        ("age + charge",
            ["Age", "ChargeDegree"]),
        ("full criminal history",
            ["Age", "ChargeDegree", "JuvFelony", "JuvMisd", "Priors"]),
    ]
    ate_table = staged_backdoor_ate(df, "Race", "Score", stages)
    print(ate_table.to_string(index=False))
    ate_table.to_csv("results/compas_ate.csv", index=False)

    # 6. Side-by-side: correlation-based vs causal estimates
    print("\n=== Correlation-based vs causal estimates for Race -> Score ===")
    dir_score = disparate_impact_ratio(df, "Race", "Score")
    sp_score = statistical_parity(df, "Race", "Score")
    ate_full, _ = backdoor_ate(df, "Race", "Score",
                               ["Age", "ChargeDegree", "JuvFelony", "JuvMisd", "Priors"])
    print(f"  DIR (Score)              = {dir_score:.3f}")
    print(f"  Statistical parity (Score) = {sp_score:+.3f}")
    print(f"  Raw ATE (no controls)    = {ate_table.iloc[0]['ATE']:+.3f}")
    print(f"  ATE (full controls)      = {ate_full:+.3f}")
    if results.get("DirectLiNGAM") is not None:
        beta_dl = results["DirectLiNGAM"].get_coefficient("Race", "Score")
        print(f"  DirectLiNGAM beta_hat    = {beta_dl:+.4f}")
    if results.get("ICA-LiNGAM") is not None:
        beta_ica = results["ICA-LiNGAM"].get_coefficient("Race", "Score")
        print(f"  ICA-LiNGAM   beta_hat    = {beta_ica:+.4f}")

    # 7. Side-by-side comparison figure (correlation vs causal panel plot)
    print("\n=== Generating correlation-vs-causal comparison figure ===")
    correlation_metrics = {
        "COMPAS Score\nDisparate Impact\n(\u22650.8 = fair)":
            base["DIR_score"],
        "COMPAS Score\nStat. Parity\n(=0 = fair)":
            base["stat_parity_score"],
        "Recidivism\nDisparate Impact":
            base["DIR_recidivism"],
    }
    causal_estimates = {
        "ATE Race\u2192Score\n(no controls)":   ate_table.iloc[0]["ATE"],
        "ATE Race\u2192Score\n(full controls)": ate_full,
    }
    # Add LiNGAM bars only when those algorithms succeeded. Show BOTH
    # DirectLiNGAM and ICA-LiNGAM as separate bars when they disagree --
    # the paper figure collapsed them into one because they happened to
    # agree on the original data; with current causal-learn / NumPy the
    # estimates can differ, and showing both is the honest representation.
    if results.get("DirectLiNGAM") is not None:
        beta_dl = results["DirectLiNGAM"].get_coefficient("Race", "Score")
        if beta_dl is not None:
            causal_estimates["DirectLiNGAM \u03b2\nRace\u2192Score"] = beta_dl
    if results.get("ICA-LiNGAM") is not None:
        beta_ica = results["ICA-LiNGAM"].get_coefficient("Race", "Score")
        if beta_ica is not None:
            causal_estimates["ICA-LiNGAM \u03b2\nRace\u2192Score"] = beta_ica

    plot_correlation_vs_causal(
        correlation_metrics=correlation_metrics,
        causal_estimates=causal_estimates,
        save_path="figures/compas_corr_vs_causal",
    )

    print("\nAll outputs written to results/ and figures/")


if __name__ == "__main__":
    main()
