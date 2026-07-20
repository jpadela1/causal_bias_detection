"""
bootstrap_ci.py
===============
Compute bootstrap 95% confidence intervals for the three COMPAS estimates
that anchor the "convergent evidence" claim.
    1. Backdoor-adjusted DE (OLS, Race -> Score with full controls)
    2. DirectLiNGAM beta-hat (Race -> Score edge coefficient)
    3. ICA-LiNGAM beta-hat (Race -> Score edge coefficient)

Method: nonparametric bootstrap. Resample n=5,278 rows with replacement
n_boot times, recompute each estimate on each resample, take the 2.5th and
97.5th percentiles. We use 1,000 resamples — enough for stable 95% CIs.
(More gives slightly tighter percentile estimates but yields diminishing
returns; 10,000 would take ~10x longer and barely change the CI.)

Reproducibility: each bootstrap iteration uses a deterministic seed
derived from a fixed base seed, so the CIs are exactly reproducible.

Run:
    python bootstrap_ci.py
Output:
    results/compas_bootstrap_ci.csv
    Console table summarizing point estimates and 95% CIs

Runtime: ~3-5 minutes on a laptop for 1,000 iterations.
"""
from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from compas_analysis import load_compas, preprocess_compas

# -- Configuration -----------------------------------------------------------
N_BOOT = 1000        # 1000 is the standard for 95% CIs; 200 is the minimum
BASE_SEED = 42       # for reproducibility
CI_LEVEL = 95        # report the 2.5th and 97.5th percentiles
ALPHA = (100 - CI_LEVEL) / 2  # = 2.5

# Backdoor adjustment set used "full controls"
FULL_CONTROLS = ["Age", "ChargeDegree", "JuvFelony", "JuvMisd", "Priors"]


# -- Estimators --------------------------------------------------------------

def estimate_ate(df: pd.DataFrame) -> float:
    """OLS backdoor-adjusted ATE of Race on Score with full controls.

    Returns the coefficient on Race from the regression
        Score ~ Race + Age + ChargeDegree + JuvFelony + JuvMisd + Priors
    """
    X = df[["Race"] + FULL_CONTROLS].astype(float)
    X = sm.add_constant(X)
    y = df["Score"].astype(float)
    model = sm.OLS(y, X).fit()
    return float(model.params["Race"])


def estimate_directlingam_beta(df: pd.DataFrame) -> float:
    """DirectLiNGAM coefficient on the Race -> Score edge.

    Returns the entry of LiNGAM's adjacency matrix corresponding to the
    Race -> Score edge. If the algorithm orients
    the edge the wrong way or omits it, returns 0.0 so the bootstrap
    distribution captures that as a real possibility (rather than
    silently dropping such iterations and shrinking the CI).
    """
    from lingam import DirectLiNGAM
    vars_ = ["Race", "Sex", "Age", "JuvFelony", "JuvMisd",
             "Priors", "ChargeDegree", "Score", "Recidivism"]
    X = df[vars_].astype(float).values
    model = DirectLiNGAM()
    model.fit(X)
    # adjacency_matrix_[i, j] is the coefficient of variable j in equation for i
    # So Race -> Score is adjacency_matrix_[Score_index, Race_index]
    race_idx, score_idx = vars_.index("Race"), vars_.index("Score")
    return float(model.adjacency_matrix_[score_idx, race_idx])


def estimate_icalingam_beta(df: pd.DataFrame) -> float:
    """ICA-LiNGAM coefficient on the Race -> Score edge. Same convention."""
    from lingam import ICALiNGAM
    vars_ = ["Race", "Sex", "Age", "JuvFelony", "JuvMisd",
             "Priors", "ChargeDegree", "Score", "Recidivism"]
    X = df[vars_].astype(float).values
    model = ICALiNGAM(random_state=BASE_SEED)
    model.fit(X)
    race_idx, score_idx = vars_.index("Race"), vars_.index("Score")
    return float(model.adjacency_matrix_[score_idx, race_idx])


# -- Bootstrap loop ----------------------------------------------------------

def bootstrap_one(df: pd.DataFrame, estimator: Callable, n_boot: int,
                  base_seed: int, label: str) -> np.ndarray:
    """Run a single estimator over n_boot bootstrap resamples of df."""
    rng = np.random.default_rng(base_seed)
    n = len(df)
    results = np.empty(n_boot)
    t0 = time.time()
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)        # sample with replacement
        sample = df.iloc[idx].reset_index(drop=True)
        try:
            results[b] = estimator(sample)
        except Exception as e:
            # Rare: LiNGAM occasionally fails on a pathological resample.
            # Record NaN; reported below in the summary.
            results[b] = np.nan
            if b < 3:  # print first few failures only
                print(f"  [{label}] iter {b}: {type(e).__name__}: {e}")
        if (b + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            eta = (n_boot - b - 1) / rate
            print(f"  [{label}] {b+1}/{n_boot}  "
                  f"({rate:.1f} iter/s, ~{eta:.0f}s remaining)")
    return results


def summarize(label: str, point_estimate: float,
              boot_samples: np.ndarray) -> dict:
    """Compute the bootstrap 95% CI and a few diagnostics."""
    finite = boot_samples[np.isfinite(boot_samples)]
    n_failed = int(np.sum(~np.isfinite(boot_samples)))
    lo, hi = np.percentile(finite, [ALPHA, 100 - ALPHA])
    return {
        "estimator":      label,
        "point_estimate": round(point_estimate, 4),
        "boot_mean":      round(float(finite.mean()), 4),
        "boot_se":        round(float(finite.std(ddof=1)), 4),
        "ci_lo":          round(float(lo), 4),
        "ci_hi":          round(float(hi), 4),
        "ci_excludes_0":  bool(lo > 0 or hi < 0),
        "n_iter":         len(boot_samples),
        "n_failed":       n_failed,
    }


# -- Main --------------------------------------------------------------------

def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 72)
    print(f"BOOTSTRAP {CI_LEVEL}% CONFIDENCE INTERVALS — COMPAS")
    print(f"  n_boot = {N_BOOT}, base_seed = {BASE_SEED}")
    print("=" * 72)

    # Load + preprocess (same pipeline as main_compas.py)
    raw = load_compas(use_local_only=False)
    df = preprocess_compas(raw, restrict_to_aa_caucasian=True)
    print(f"Sample: n={len(df)} (AA + Caucasian)")

    # Point estimates on the original sample
    print("\n--- Computing point estimates on original sample ---")
    pt_ate = estimate_ate(df)
    pt_dl  = estimate_directlingam_beta(df)
    pt_ica = estimate_icalingam_beta(df)
    print(f"  DE (full controls):      {pt_ate:+.4f}")
    print(f"  DirectLiNGAM beta_hat:    {pt_dl:+.4f}")
    print(f"  ICA-LiNGAM beta_hat:      {pt_ica:+.4f}")

    # Bootstrap each estimator
    print(f"\n--- Bootstrapping DE ({N_BOOT} resamples) ---")
    boot_ate = bootstrap_one(df, estimate_ate,
                             N_BOOT, BASE_SEED + 1, "ATE")

    print(f"\n--- Bootstrapping DirectLiNGAM ({N_BOOT} resamples) ---")
    boot_dl = bootstrap_one(df, estimate_directlingam_beta,
                            N_BOOT, BASE_SEED + 2, "DirectLiNGAM")

    print(f"\n--- Bootstrapping ICA-LiNGAM ({N_BOOT} resamples) ---")
    boot_ica = bootstrap_one(df, estimate_icalingam_beta,
                             N_BOOT, BASE_SEED + 3, "ICA-LiNGAM")

    # Summarize
    rows = [
        summarize("DE (full controls)",    pt_ate, boot_ate),
        summarize("DirectLiNGAM β̂ Race→Score", pt_dl,  boot_dl),
        summarize("ICA-LiNGAM β̂ Race→Score",   pt_ica, boot_ica),
    ]
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print(f"BOOTSTRAP {CI_LEVEL}% CONFIDENCE INTERVALS")
    print("=" * 72)
    print(summary.to_string(index=False))
    summary.to_csv("results/compas_bootstrap_ci.csv", index=False)
    print(f"\nWrote results/compas_bootstrap_ci.csv")

    # Convergent evidence check: do the three CIs overlap?
    print("\n--- Convergent evidence check ---")
    intervals = [
        ("DE",          rows[0]["ci_lo"], rows[0]["ci_hi"]),
        ("DirectLiNGAM", rows[1]["ci_lo"], rows[1]["ci_hi"]),
        ("ICA-LiNGAM",   rows[2]["ci_lo"], rows[2]["ci_hi"]),
    ]
    for name, lo, hi in intervals:
        sig = "***" if (lo > 0 or hi < 0) else "ns "
        print(f"  {sig} {name:14s} 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print()
    print("  Three estimators agree in sign?",
          all(rows[i]["point_estimate"] > 0 for i in range(3)) or
          all(rows[i]["point_estimate"] < 0 for i in range(3)))
    print("  All three CIs exclude zero?    ",
          all(r["ci_excludes_0"] for r in rows))


if __name__ == "__main__":
    main()
