"""
main_compas.py
==============
Study 2: COMPAS recidivism analysis.

Pipeline:
  1. Load ProPublica COMPAS two-year CSV (downloads on first run, then caches).
  2. Apply ProPublica's standard preprocessing filters; restrict to
     African-American + Caucasian so the binary Race=1/0 encoding matches
     the ProPublica article.
  3. Compute baseline correlation-based fairness metrics, including BOTH the
     descriptive mean-Score ratio AND the proper selection-rate DIR (the one
     that can legitimately be compared against the 4/5-rule threshold).
  4. Print ProPublica's contingency-table numbers (FP/FN rates, PPV, NPV) on
     the n=7,214 sample so the paper anchors to the literature.
  5. Run all six causal discovery algorithms; tabulate Race -> Score and
     Race -> Priors detections.
  6. Plot individual DAGs and the algorithm grid (shared node positions).
  7. Compute backdoor-adjusted ATE with progressive control sets.
  8. Build the correlation-vs-causal comparison figure with corrected labels.

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
    per_race_breakdown,
    propublica_contingency,
    plot_correlation_vs_causal,
)
from causal_discovery import run_all, report_convention
from ate_estimation import (
    backdoor_ate,
    staged_backdoor_ate,
)
from visualization import (
    plot_discovery_result,
    plot_grid,
    compute_shared_pos,
    DEFAULT_ROLES_COMPAS,
)
def _make_dag_title(name: str, n: int, res, include_n: bool = True) -> str:
    """Build a per-DAG title with optional n and (when available) the
    Race->Score β̂.
    Parameters
    ----------
    include_n : bool, default True
        Set False for grid panels — n is shown once in the grid suptitle
        instead, avoiding the redundant repetition across all six panels.
    """
    beta = None
    if res is not None and res.coef_matrix is not None:
        try:
            beta = res.get_coefficient("Race", "Score")
        except Exception:
            beta = None

    parts = []
    if include_n:
        parts.append(f"n={n:,}")
    if beta is not None:
        parts.append(r"$\hat{\beta}$" + f"={beta:+.4f}")

    if parts:
        return f"{name}\n" + ", ".join(parts)
    return name


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

    # ---------------------------------------------------------------------
    # 1. Load + preprocess
    # ---------------------------------------------------------------------
    raw = load_compas()
    df  = preprocess_compas(raw, restrict_to_aa_caucasian=True)
    print(f"\nPreprocessed COMPAS shape: {df.shape}  "
          f"(filtered to African-American + Caucasian)")

    # ---------------------------------------------------------------------
    # 2. Per-race breakdown on the standard
    #    ProPublica-filtered sample BEFORE the AA/Cau restriction.
    # ---------------------------------------------------------------------
    print("\n=== Per-race breakdown (Table III; ProPublica filters, all races) ===")
    print(per_race_breakdown(raw).to_string(index=False))

    # ---------------------------------------------------------------------
    # 3. Baseline disparities (correlation-based)
    # ---------------------------------------------------------------------
    print("\n=== Baseline disparities (correlation-based) ===")
    base = baseline_disparities(df)
    for k, v in base.items():
        print(f"  {k:30s} {v}")

    # Highlight the contrast between the two ways of computing "DIR"
    print("\n  --- Score disparity, two ways ---")
    print(f"  Mean-ratio (paper's old metric, NOT 4/5-rule comparable): "
          f"{base['mean_score_ratio']}")
    print(f"  Selection-rate DIR P(Score>=5|AA)/P(Score>=5|Cau): "
          f"{base['DIR_score_selection_rate']}  "
          f"(0.80 threshold applies to this one)")

    # ---------------------------------------------------------------------
    # 4. ProPublica's contingency-table numbers on n=7,214 sample
    # ---------------------------------------------------------------------
    print("\n=== ProPublica contingency-table reproduction (n=7,214 sample) ===")
    print("  (These should match the published 44.85/27.99 and 23.45/47.72)")
    for race in ["African-American", "Caucasian"]:
        c = propublica_contingency(raw, race)
        print(f"  {race:18s} n={c['n']:5d}  "
              f"FP={c['FP_rate']:5.2f}%  FN={c['FN_rate']:5.2f}%  "
              f"PPV={c['PPV']:.2f}  NPV={c['NPV']:.2f}")

    # ---------------------------------------------------------------------
    # R² of each variable as a function of all others.
    # This documents the empirical basis for the GRaSP failure-mechanism the COMPAS Score has the highest R² (most
    # explained by other features), which GRaSP's BIC residual-variance
    # ordering misinterprets as exogeneity.
    # ---------------------------------------------------------------------
    print("\n=== R² of each variable regressed on all others ===")
    print("(Highest R² = most endogenous; GRaSP misclassifies this as exogenous)")
    from sklearn.linear_model import LinearRegression
    vars_ = ["Race", "Sex", "Age", "JuvFelony", "JuvMisd",
             "Priors", "ChargeDegree", "Score", "Recidivism"]
    r2_rows = []
    for target in vars_:
        predictors = [v for v in vars_ if v != target]
        r2 = LinearRegression().fit(
            df[predictors].astype(float), df[target].astype(float)
        ).score(df[predictors].astype(float), df[target].astype(float))
        r2_rows.append({"variable": target, "R2": round(r2, 4)})
    r2_df = pd.DataFrame(r2_rows).sort_values("R2", ascending=False)
    print(r2_df.to_string(index=False))
    r2_df.to_csv("results/compas_r2.csv", index=False)
    # ---------------------------------------------------------------------
    # 5. Causal discovery
    # ---------------------------------------------------------------------
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

    # Tabulate Race->Score and Race->Priors detections
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

    # ---------------------------------------------------------------------
    # 6. Shared node positions + DAG figures (individual + grid)
    # ---------------------------------------------------------------------
    print("\n--- Computing shared node positions (fixed layout: outcomes right) ---")
    shared_pos = compute_shared_pos(results)
    print(f"  Positions computed for: {list(shared_pos.keys())}")

    flagged = [("Race", "Score"), ("Race", "Priors")]

    print("\n--- Plotting individual DAGs ---")
    for name, res in results.items():
        if res is None:
            continue
        plot_discovery_result(
            res,
            title=_make_dag_title(name, len(df), res),
            flagged_edges=flagged,
            node_roles=DEFAULT_ROLES_COMPAS,
            pos=shared_pos,
            save_path=f"figures/compas_{name.replace('-', '')}.pdf",
        )
        plt.close()

    print("\n--- Plotting algorithm grid ---")
    panel_titles = {
        name: _make_dag_title(name, len(df), res, include_n=False)
        for name, res in results.items() if res is not None
    }
    plot_grid(
        results,
        flagged_edges=flagged,
        node_roles=DEFAULT_ROLES_COMPAS,
        title=f"All algorithms on COMPAS, n={len(df):,}",
        pos=shared_pos,
        panel_titles=panel_titles,
        save_path="figures/compas_grid.pdf",
    )
    plt.close("all")

    # ---------------------------------------------------------------------
    # 7. Backdoor-adjustment ATE
    # ---------------------------------------------------------------------
    print("\n=== Backdoor DE: Race -> Score ===")
    print("Formula: DE = E_Z[ E[Y|T=1,Z] - E[Y|T=0,Z] ]")
    print("Under linearity: DE_hat = OLS coefficient on Race after controlling for Z")

    stages = [
        ("no controls",           []),
        ("age + charge",          ["Age", "ChargeDegree"]),
        ("full criminal history", ["Age", "ChargeDegree",
                                   "JuvFelony", "JuvMisd", "Priors"]),
    ]
    ate_table = staged_backdoor_ate(df, "Race", "Score", stages)
    print(ate_table.to_string(index=False))
    ate_table.to_csv("results/compas_ate.csv", index=False)

    # ---------------------------------------------------------------------
    # 8. Side-by-side: correlation-based vs causal estimates
    # ---------------------------------------------------------------------
    print("\n=== Correlation-based vs causal estimates for Race -> Score ===")
    ate_full, _ = backdoor_ate(
        df, "Race", "Score",
        ["Age", "ChargeDegree", "JuvFelony", "JuvMisd", "Priors"],
    )
    print(f"  Mean-Score ratio (descriptive)    = {base['mean_score_ratio']:.4f}")
    print(f"  Selection-rate DIR (proper)       = {base['DIR_score_selection_rate']:.4f}")
    print(f"  DIR_recidivism (rate ratio)       = {base['DIR_recidivism']:.4f}")
    print(f"  Statistical parity (Score)        = {base['stat_parity_score']:+.4f}")
    print(f"  Raw DE (no controls)             = {ate_table.iloc[0]['ATE']:+.4f}")
    print(f"  DE (full controls)               = {ate_full:+.4f}")

    beta_dl = beta_ica = None
    if results.get("DirectLiNGAM") is not None:
        beta_dl = results["DirectLiNGAM"].get_coefficient("Race", "Score")
        print(f"  DirectLiNGAM beta_hat             = {beta_dl:+.4f}")
    if results.get("ICA-LiNGAM") is not None:
        beta_ica = results["ICA-LiNGAM"].get_coefficient("Race", "Score")
        print(f"  ICA-LiNGAM   beta_hat             = {beta_ica:+.4f}")

    # -----------------------------------------------------------------
    # Correlation vs causal comparison figure
    # -----------------------------------------------------------------
    print("\n=== Generating correlation-vs-causal comparison figure ===")
    correlation_metrics = {
        "COMPAS Score\nSelection-rate DIR\n(\u22650.8 = fair)":
            base["DIR_score_selection_rate"],
        "Recidivism\nDisparate Impact\n(rate ratio)":
            base["DIR_recidivism"],
        "COMPAS Score\nMean ratio\n(descriptive*)":
            base["mean_score_ratio"],
    }
    causal_estimates = {
        "DE Race\u2192Score\n(no controls)":   ate_table.iloc[0]["ATE"],
        "DE Race\u2192Score\n(full controls)": ate_full,
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
    print("\n*Note: 'Mean ratio' in the figure is shown for transparency but the")
    print(" 4/5-rule threshold (dashed line at 0.8) only applies to selection-")
    print(" rate metrics, not means of ordinal scores. The selection-rate DIR")
    print(" is the legally meaningful one.")


if __name__ == "__main__":
    main()
