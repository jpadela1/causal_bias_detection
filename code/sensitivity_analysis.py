"""
sensitivity_analysis.py
=======================
Reproduce the sensitivity-analysis grid from Section V of the paper.

Sweeps:
    bias coefficient  beta in {0.00, 0.05, 0.10, 0.15, 0.20, 0.25}   (sign-flipped: paper uses negative beta)
    sample size       n    in {1000, 50000}                          (configurable)
    repetitions       20 random seeds per cell                       (configurable)

For each (beta, n, seed) cell we record:
    - whether each algorithm detects the Race -> Loan edge
    - the SHD against ground truth
    - the LiNGAM beta-hat (when applicable)

Note: n=50000 with 20 seeds and 6 algorithms takes a while; the script
exposes `sample_sizes`, `betas`, and `n_repeats` so you can shrink the grid
during development.
"""
from __future__ import annotations

import os
import time
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from synthetic_data import generate_loan_data
from causal_discovery import (
    ALGORITHMS,
    DiscoveryResult,
    structural_hamming_distance,
)


GROUND_TRUTH_EDGES_BIASED = [
    ("Race", "ZIP"),
    ("Race", "Income"),
    ("Race", "Loan"),
    ("Gender", "Education"),
    ("Gender", "Income"),
    ("Education", "Income"),
    ("Education", "CreditSc"),
    ("ZIP", "CreditSc"),
    ("Income", "CreditSc"),
    ("Income", "Loan"),
    ("CreditSc", "Loan"),
]
GROUND_TRUTH_EDGES_UNBIASED = [e for e in GROUND_TRUTH_EDGES_BIASED if e != ("Race", "Loan")]


def run_sensitivity_grid(
    betas: List[float] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25),
    sample_sizes: List[int] = (1000, 5000, 10000, 50000),
    n_repeats: int = 20,
    base_seed: int = 1000,
    out_csv: str = "results/sensitivity_results.csv",
) -> pd.DataFrame:
    """Run the full sensitivity grid; write per-run records to CSV.

    Each row is one (algorithm, beta, n, seed) cell.
    """
    records = []
    total = len(betas) * len(sample_sizes) * n_repeats * len(ALGORITHMS)
    pbar = tqdm(total=total, desc="Sensitivity grid")

    for beta in betas:
        # NOTE: paper uses NEGATIVE beta; here `beta` is the magnitude. We
        # generate data with `-beta` so the planted edge points the right way.
        beta_signed = -beta
        truth = GROUND_TRUTH_EDGES_BIASED if beta != 0.0 else GROUND_TRUTH_EDGES_UNBIASED
        for n in sample_sizes:
            for rep in range(n_repeats):
                seed = base_seed + rep
                data = generate_loan_data(n=n, beta=beta_signed, seed=seed)

                for alg_name, fn in ALGORITHMS.items():
                    t0 = time.perf_counter()
                    try:
                        res: DiscoveryResult = fn(data)
                        elapsed = time.perf_counter() - t0
                        detected = res.has_directed_edge("Race", "Loan")
                        shd = structural_hamming_distance(res, truth)
                        beta_hat = res.get_coefficient("Race", "Loan")
                    except Exception as e:
                        elapsed = time.perf_counter() - t0
                        detected = False
                        shd = np.nan
                        beta_hat = None
                        print(f"[WARN] {alg_name} failed on (beta={beta}, n={n}, seed={seed}): {e}")

                    records.append({
                        "algorithm": alg_name,
                        "beta": beta,
                        "n": n,
                        "seed": seed,
                        "detected_race_to_loan": int(bool(detected)),
                        "shd": shd,
                        "beta_hat": beta_hat,
                        "wall_time_s": round(elapsed, 3),
                    })
                    pbar.update(1)

    pbar.close()
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")
    return df


def aggregate_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-cell results: detection rate + mean SHD."""
    g = df.groupby(["algorithm", "beta", "n"])
    summary = g.agg(
        detection_rate=("detected_race_to_loan", "mean"),
        mean_shd=("shd", "mean"),
        std_shd=("shd", "std"),
        mean_beta_hat=("beta_hat", "mean"),
    ).reset_index()
    summary["mean_shd"] = summary["mean_shd"].round(2)
    summary["std_shd"] = summary["std_shd"].round(2)
    summary["detection_rate"] = summary["detection_rate"].round(2)
    return summary


def plot_sensitivity_heatmap(
    summary: pd.DataFrame,
    save_path: str = "figures/sensitivity_grid.png",
):
    """Recreate Figure 2 from the paper: detection rate as bar height, SHD as annotation."""
    import matplotlib.pyplot as plt

    algorithms = list(ALGORITHMS.keys())
    sample_sizes = sorted(summary["n"].unique())
    betas = sorted(summary["beta"].unique())

    fig, axes = plt.subplots(
        nrows=len(sample_sizes),
        ncols=len(algorithms),
        figsize=(2.5 * len(algorithms), 2.2 * len(sample_sizes)),
        sharex=True, sharey=True,
    )
    if len(sample_sizes) == 1:
        axes = axes.reshape(1, -1)

    for r, n in enumerate(sample_sizes):
        for c, alg in enumerate(algorithms):
            ax = axes[r, c]
            sub = summary[(summary["algorithm"] == alg) & (summary["n"] == n)].sort_values("beta")
            heights = sub["detection_rate"].values
            colors = ["#2ca02c" if h >= 0.95 else "#7fbf7b" if h >= 0.5 else "#d9d9d9"
                      for h in heights]
            ax.bar(range(len(sub)), heights, color=colors, edgecolor="black", linewidth=0.5)
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(
                    i, 0.05, f"{row['mean_shd']:.0f}±{row['std_shd']:.0f}",
                    ha="center", va="bottom", fontsize=6,
                )
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels([f"{b:.2f}" for b in sub["beta"]], fontsize=6, rotation=45)
            if r == 0:
                ax.set_title(alg, fontsize=9, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"n={n:,}\n\nDetection rate", fontsize=8)
            if r == len(sample_sizes) - 1:
                ax.set_xlabel("β", fontsize=8)

    fig.suptitle(
        "Sensitivity Analysis: Race→Loan Detection Rate and SHD\n"
        "(bar height = detection rate; annotation = mean SHD ± std)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    from visualization import save_figure_dual_format
    save_figure_dual_format(fig, save_path)
    plt.close(fig)


if __name__ == "__main__":
    # Small demo grid; expand betas/sample_sizes/n_repeats for the full paper grid.
    #   betas: List[float] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25),
    # sample_sizes: List[int] = (1000, 5000, 10000, 50000),
    df = run_sensitivity_grid(
        betas=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
        sample_sizes=[1000, 5000, 10000, 50000],
        n_repeats=10,
        out_csv="results/sensitivity_results_demo.csv",
    )
    summary = aggregate_grid(df)
    print("\nAggregated summary:")
    print(summary.to_string(index=False))
    summary.to_csv("results/sensitivity_summary_demo.csv", index=False)
    plot_sensitivity_heatmap(summary, save_path="figures/sensitivity_grid_demo.png")
