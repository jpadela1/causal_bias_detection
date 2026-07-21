"""
sensitivity_analysis.py
=======================

Uses:
    bias coefficient  beta in {0.00, 0.05, 0.10, 0.15, 0.20, 0.25}   (sign-flipped: Study 1 uses negative beta)
    sample size       n    in {1000, 5000, 10000, 50000}             (configurable)
    repetitions       20 random seeds per cell                       (configurable)

For each (beta, n, seed) cell we record:
    - whether each algorithm detects the Race -> Loan edge
    - the SHD against ground truth
    - the LiNGAM beta-hat (when applicable)

Note: n=50000 with 20 seeds and 6 algorithms takes a while; the script
exposes `sample_sizes`, `betas`, and `n_repeats` so you can shrink the grid
during development.

Detection convention: `detected_race_to_loan` always records whether the
Race -> Loan edge is PRESENT in the recovered graph. At beta = 0.00 the edge
is absent from the ground truth, so the recorded rate at that column is the
false-positive rate (lower is better), not a "correctly absent" rate.
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
# Main experimental setting reported in Table II; highlighted in the figure.
MAIN_BETA = 0.15

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
        # NOTE: Here `beta` is the magnitude. We
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
                        #res: DiscoveryResult = fn(data) # fix for GRaSP seeding and algorithm stability
                        res: DiscoveryResult = (
                            fn(data, seed=seed) if alg_name == "GRaSP" else fn(data)
                        )
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

def _fmt_shd(mean: float, std: float) -> str:
    """One decimal when either component is sub-integer, else integer.

    '.0f' on (0.35, 0.49) renders '0±0', which reads as perfect structural
    recovery with zero variance. Neither holds: at n=5,000 the LiNGAM cells
    have 13 of 20 seeds at SHD 0 and 7 at SHD 1. Printing one decimal keeps
    that spread visible and consistent with Table II, where the pinned-seed
    run reports SHD = 1.
    """
    if pd.isna(mean) or pd.isna(std):
        return "n/a"
    if mean < 1 or std < 1:
        return f"{mean:.1f}±{std:.1f}"
    return f"{mean:.0f}±{std:.0f}"

def plot_sensitivity_heatmap(
    summary: pd.DataFrame,
    save_path: str = "figures/sensitivity_grid.png",
):
    """Detection rate as bar height, SHD as annotation."""
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

            bars = ax.bar(
                range(len(sub)), heights,
                width=0.72, color="#2e7d4f",
                edgecolor="#1b1b1b", linewidth=0.4,
            )
            # colour intensity tracks detection rate; main setting gets a red outline
            for bar, h, b in zip(bars, heights, sub["beta"]):
                bar.set_alpha(0.25 + 0.75 * float(h))
                if abs(float(b) - MAIN_BETA) < 1e-9:
                    bar.set_edgecolor("#b00020")
                    bar.set_linewidth(1.3)

            # annotation sits above its own bar and runs vertically, so labels
            # no longer collide along a shared baseline
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(
                    i, min(float(row["detection_rate"]) + 0.04, 0.72),
                    _fmt_shd(row["mean_shd"], row["std_shd"]),
                    ha="center", va="bottom", rotation=90,
                    fontsize=6.4, color="#222222",
                )

            ax.set_ylim(0, 1.0)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels([f"{b:.2f}" for b in sub["beta"]], fontsize=6.5, rotation=90)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55)
            ax.set_axisbelow(True)

            if r == 0:
                ax.set_title(alg, fontsize=10, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(f"n={n:,}\nDetection rate", fontsize=8.5)
            if r == len(sample_sizes) - 1:
                ax.set_xlabel("β", fontsize=9)

    fig.suptitle(
        "Sensitivity Analysis: Race→Loan Detection Rate and SHD",
        fontsize=13, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.955,
        "bar height = detection rate; annotation = mean SHD ± std over "
        "20 seeds (one decimal where mean or std < 1); "
        "red outline = main setting β = 0.15",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    from visualization import save_figure_dual_format
    save_figure_dual_format(fig, save_path)
    plt.close(fig)

if __name__ == "__main__":
    df = run_sensitivity_grid(
        betas=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
        sample_sizes=[1000, 5000, 10000, 50000],
        n_repeats=20,
        base_seed=1000,
        out_csv="results/sensitivity_results.csv",
    )
    summary = aggregate_grid(df)
    print("\nAggregated summary:")
    print(summary.to_string(index=False))
    summary.to_csv("results/sensitivity_summary.csv", index=False)
    plot_sensitivity_heatmap(summary, save_path="figures/sensitivity_grid.png")
