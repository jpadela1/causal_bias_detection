"""
main_sensitivity.py
===================
Reproduce Section V (Sensitivity Analysis) of the paper.

By default this runs a SMALL grid (faster). Edit the lists below to expand to
the full paper grid:
    betas        = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    sample_sizes = [1000, 50000]
    n_repeats    = 20
WARNING: the full grid runs hundreds of LiNGAM/GES fits and can take >1 hour
on a laptop.

Outputs:
    results/sensitivity_results.csv     (one row per algorithm-cell-seed)
    results/sensitivity_summary.csv     (detection rate + mean SHD per cell)
    figures/sensitivity_grid.png        (Figure 2 reproduction)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from sensitivity_analysis import (
    run_sensitivity_grid,
    aggregate_grid,
    plot_sensitivity_heatmap,
)


def main():
    # ---- Adjust grid here ------------------------------------------------
    betas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    sample_sizes = [1000, 5000, 10000, 50000]   # add 50000 for full paper grid (slow!)
    n_repeats = 20                  # paper uses 20
    # ----------------------------------------------------------------------

    df = run_sensitivity_grid(
        betas=betas,
        sample_sizes=sample_sizes,
        n_repeats=n_repeats,
        out_csv="results/sensitivity_results.csv",
    )

    summary = aggregate_grid(df)
    summary.to_csv("results/sensitivity_summary.csv", index=False)
    print("\n=== Aggregated sensitivity summary ===")
    print(summary.to_string(index=False))

    plot_sensitivity_heatmap(
        summary, save_path="figures/sensitivity_grid.png"
    )


if __name__ == "__main__":
    main()
