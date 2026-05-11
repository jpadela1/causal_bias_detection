"""
main_synthetic.py
=================
Reproduce Study 1 (Section IV): synthetic loan-approval analysis.


From the sensitivity dataset and to provide an equivalent comparison data using a similar size ground truth of
5,000


Run this first. It will:
  1. Generate Dataset A (biased, beta=-0.15) and Dataset B (unbiased, beta=0).
  2. Run all six causal discovery algorithms on each dataset.
  3. Print a Table II-style summary (SHD, Race->Loan detection, beta_hat).
  4. Save individual + grid DAG figures to figures/.
  5. Compute backdoor ATE for Race -> Loan with a sequence of adjustment sets.

Outputs go to:
    results/synthetic_summary.csv
    figures/synthetic_*.png
"""
from __future__ import annotations

import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from synthetic_data import generate_paired_datasets, plot_ground_truth_dag
from causal_discovery import run_all, structural_hamming_distance, report_convention
from ate_estimation import staged_backdoor_ate, disparate_impact_ratio
from visualization import plot_discovery_result, plot_grid, DEFAULT_ROLES_LOAN
from sensitivity_analysis import (
    GROUND_TRUTH_EDGES_BIASED,
    GROUND_TRUTH_EDGES_UNBIASED,
)


def _make_dag_title(name: str, n: int, res, include_n: bool = True) -> str:
    """Build a per-DAG title with optional n and (when available) the
    Race->Loan β̂. For Study 1 the audited edge is Race->Loan."""
    beta = None
    if res is not None and res.coef_matrix is not None:
        try:
            beta = res.get_coefficient("Race", "Loan")
        except Exception:
            beta = None

    parts = []
    if include_n:
        parts.append(f"n={n:,}")
    if beta is not None:
        parts.append(r"$\hat{\beta}$" + f"={beta:+.3f}")

    if parts:
        return f"{name}\n" + ", ".join(parts)
    return name


def summarize_dataset(name: str, results: dict, ground_truth) -> pd.DataFrame:
    rows = []
    for alg, res in results.items():
        if res is None:
            rows.append({"dataset": name, "algorithm": alg, "Race->Loan": "ERROR",
                         "SHD": None, "beta_hat": None, "lat_conf": False})
            continue
        rows.append({
            "dataset": name,
            "algorithm": alg,
            "Race->Loan": "YES" if res.has_directed_edge("Race", "Loan") else "no",
            "SHD": structural_hamming_distance(res, ground_truth),
            "beta_hat": (round(res.get_coefficient("Race", "Loan"), 4)
                         if res.coef_matrix is not None else None),
            "lat_conf": bool(res.bidirected_edges),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("=" * 72)
    print("STUDY 1: SYNTHETIC LOAN-APPROVAL DATASET")
    print("=" * 72)

    # Detect causal-learn's edge-encoding convention before any algorithm
    # runs. This protects against version drift in the underlying library:
    # if the convention differs from what the docs say, the auto-detector
    # picks the right one so PC/FCI/GES/GRaSP plots show the correct
    # arrow directions. LiNGAM is unaffected (uses a different code path).
    print()
    report_convention()

    # Render the ground-truth DAG up front so reviewers can compare every
    # algorithm's recovered graph against the SCM that generated the data.
    print("\n--- Rendering ground-truth DAG ---")
    plot_ground_truth_dag(
        save_path="figures/synthetic_ground_truth",
        show_both_versions=True,
    )

    # ----------------------------------------------------------------------
    # Generate paired datasets
    # ----------------------------------------------------------------------
    biased, unbiased = generate_paired_datasets(n=5000, beta_biased=-0.15, seed=42)
    n = len(biased)   # both datasets have the same size

    print(f"\nDataset A (biased)   shape: {biased.shape}")
    print(f"Dataset B (unbiased) shape: {unbiased.shape}")
    print(f"Loan approval rate by Race (biased)  : "
          f"{biased.groupby('Race')['Loan'].mean().to_dict()}")
    print(f"Loan approval rate by Race (unbiased): "
          f"{unbiased.groupby('Race')['Loan'].mean().to_dict()}")

    # ----------------------------------------------------------------------
    # Run all algorithms
    # ----------------------------------------------------------------------
    print("\n--- Running causal discovery on Dataset A (biased) ---")
    results_biased = run_all(biased)
    for name, res in results_biased.items():
        if res is None:
            continue
        print(res.summary())
        print()

    print("\n--- Running causal discovery on Dataset B (unbiased) ---")
    results_unbiased = run_all(unbiased)
    for name, res in results_unbiased.items():
        if res is None:
            continue
        print(res.summary())
        print()

    # ----------------------------------------------------------------------
    # Table II
    # ----------------------------------------------------------------------
    print("\n=== Table II: Algorithm performance on synthetic dataset ===")
    df_b = summarize_dataset("biased", results_biased, GROUND_TRUTH_EDGES_BIASED)
    df_u = summarize_dataset("unbiased", results_unbiased, GROUND_TRUTH_EDGES_UNBIASED)
    summary = pd.concat([df_b, df_u], ignore_index=True)
    print(summary.to_string(index=False))
    summary.to_csv("results/synthetic_summary.csv", index=False)
    print("  -> wrote results/synthetic_summary.csv")

    # ----------------------------------------------------------------------
    # DAG figures
    # ----------------------------------------------------------------------
    print("\n--- Plotting DAGs ---")
    flagged = [("Race", "Loan")]

    # Individual DAGs - biased dataset
    for name, res in results_biased.items():
        if res is None:
            continue
        plot_discovery_result(
            res,
            title=f"{name} on Biased Dataset (β = -0.15)\nn={n:,}"
                  + (r", $\hat{\beta}$" + f"={res.get_coefficient('Race', 'Loan'):+.3f}"
                     if res.coef_matrix is not None else ""),
            flagged_edges=flagged,
            node_roles=DEFAULT_ROLES_LOAN,
            save_path=f"figures/synthetic_biased_{name.replace('-', '')}.png",
            layout="fixed",
        )
        plt.close()

    # Individual DAGs - unbiased dataset
    for name, res in results_unbiased.items():
        if res is None:
            continue
        plot_discovery_result(
            res,
            title=f"{name} on Unbiased Dataset (β = 0)\nn={n:,}"
                  + (r", $\hat{\beta}$" + f"={res.get_coefficient('Race', 'Loan'):+.3f}"
                     if res.coef_matrix is not None else ""),
            flagged_edges=flagged,
            node_roles=DEFAULT_ROLES_LOAN,
            save_path=f"figures/synthetic_unbiased_{name.replace('-', '')}.png",
            layout="fixed",
        )
        plt.close()

    # Grid - biased dataset
    panel_titles_b = {
        name: _make_dag_title(name, n, res, include_n=False)
        for name, res in results_biased.items() if res is not None
    }
    plot_grid(
        results_biased,
        flagged_edges=flagged,
        node_roles=DEFAULT_ROLES_LOAN,
        title=f"All algorithms on Biased Dataset (β = -0.15), n={n:,}",
        save_path="figures/synthetic_biased_grid.png",
        layout="fixed",
        panel_titles=panel_titles_b,
    )

    # Grid - unbiased dataset
    panel_titles_u = {
        name: _make_dag_title(name, n, res, include_n=False)
        for name, res in results_unbiased.items() if res is not None
    }
    plot_grid(
        results_unbiased,
        flagged_edges=flagged,
        node_roles=DEFAULT_ROLES_LOAN,
        title=f"All algorithms on Unbiased Dataset (β = 0), n={n:,}",
        save_path="figures/synthetic_unbiased_grid.png",
        layout="fixed",
        panel_titles=panel_titles_u,
    )
    plt.close("all")

    # ----------------------------------------------------------------------
    # Backdoor ATE for Race -> Loan on the biased dataset
    # ----------------------------------------------------------------------
    print("\n=== Backdoor ATE: Race -> Loan (biased dataset) ===")
    print("Formula: ATE = E_Z[ E[Y|T=1,Z] - E[Y|T=0,Z] ]")
    print("Under linearity:  ATE_hat = OLS coefficient on T after adjusting for Z")
    stages = [
        ("no controls", []),
        ("+ Income", ["Income"]),
        ("+ Income, CreditSc", ["Income", "CreditSc"]),
        ("+ ZIP, Income, CreditSc", ["ZIP", "Income", "CreditSc"]),
        ("+ Gender, Education, ZIP, Income, CreditSc",
         ["Gender", "Education", "ZIP", "Income", "CreditSc"]),
    ]
    ate_table_b = staged_backdoor_ate(biased, "Race", "Loan", stages)
    ate_table_u = staged_backdoor_ate(unbiased, "Race", "Loan", stages)

    print("\nBiased dataset:")
    print(ate_table_b.to_string(index=False))
    print("\nUnbiased dataset:")
    print(ate_table_u.to_string(index=False))
    ate_table_b.to_csv("results/synthetic_ate_biased.csv", index=False)
    ate_table_u.to_csv("results/synthetic_ate_unbiased.csv", index=False)

    print(f"\nDIR (biased)   = {disparate_impact_ratio(biased,   'Race', 'Loan'):.4f}")
    print(f"DIR (unbiased) = {disparate_impact_ratio(unbiased, 'Race', 'Loan'):.4f}")
    print("\nAll outputs written to results/ and figures/")


if __name__ == "__main__":
    main()
