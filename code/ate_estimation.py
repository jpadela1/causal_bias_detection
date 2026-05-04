"""
ate_estimation.py
=================
Average Treatment Effect (ATE) estimation via Pearl's backdoor criterion.

Backdoor adjustment formula
---------------------------
For a treatment T, outcome Y, and a backdoor-admissible adjustment set Z
(satisfying Pearl's backdoor criterion w.r.t. (T, Y) in the DAG):

    ATE = E[ Y | do(T=1) ] - E[ Y | do(T=0) ]
        = E_Z[ E[Y | T=1, Z] - E[Y | T=0, Z] ]

Under a linear outcome model
    Y = beta_0 + beta_T * T + beta_Z^T * Z + epsilon

the ATE is identified by the coefficient on T:
    ATE_hat = beta_T_hat   (the OLS coefficient on T after adjusting for Z)

This is the regression-adjustment estimator. It coincides with Pearl's
backdoor adjustment under the linearity / no-interaction assumption, and is
what the paper reports in Equations (1)-(3).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _ols(X: np.ndarray, y: np.ndarray):
    """Plain OLS via numpy. Returns (coef_vector, residuals, rank, sv).
    We compute SE the closed-form way so we don't need statsmodels.

    Returns
    -------
    coef    : np.ndarray of shape (p+1,)  including intercept at index 0
    se      : np.ndarray of shape (p+1,)  matching standard errors
    """
    n, p1 = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coef
    dof = max(n - p1, 1)
    sigma2 = float(residuals @ residuals) / dof
    # (X'X)^-1 -- use pinv for numerical safety on near-collinear designs
    XtX_inv = np.linalg.pinv(X.T @ X)
    var_beta = sigma2 * np.diag(XtX_inv)
    se = np.sqrt(np.maximum(var_beta, 0.0))
    return coef, se


def backdoor_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: Sequence[str] = (),
    return_se: bool = False,
) -> Tuple[float, Optional[float]]:
    """Estimate ATE of `treatment` on `outcome` adjusting for `adjustment_set`.

    Parameters
    ----------
    data : DataFrame
    treatment : str            column name (binary or continuous)
    outcome : str              column name
    adjustment_set : iterable  backdoor-admissible variables (must be NON-DESCENDANTS
                               of `treatment` in the assumed DAG)
    return_se : bool           also return the standard error of the ATE estimate

    Returns
    -------
    (ate_hat, se_or_None)
    """
    cols = [treatment] + list(adjustment_set)
    X_df = data[cols].astype(float).copy()
    n = len(X_df)
    # Prepend intercept column
    X = np.column_stack([np.ones(n), X_df.values])
    y = data[outcome].astype(float).values

    coef, se = _ols(X, y)
    # treatment is at index 1 (after intercept)
    ate_hat = float(coef[1])
    return ate_hat, (float(se[1]) if return_se else None)


def staged_backdoor_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_stages: List[Tuple[str, Sequence[str]]],
) -> pd.DataFrame:
    """Compute ATE for a sequence of progressively larger adjustment sets.

    This reproduces the COMPAS table where the ATE shrinks as more controls
    are added (raw -> partial -> full). Each row of the returned DataFrame
    corresponds to one stage.

    Parameters
    ----------
    adjustment_stages : list of (label, [vars])
        e.g. [("no controls", []),
              ("age + charge", ["Age", "ChargeDegree"]),
              ("full history", ["Age", "ChargeDegree", "JuvFel", "JuvMisd", "Priors"])]

    Returns
    -------
    DataFrame with columns [stage, ATE, SE, n_controls, controls]
    """
    rows = []
    for label, controls in adjustment_stages:
        ate, se = backdoor_ate(data, treatment, outcome, controls, return_se=True)
        rows.append(
            {
                "stage": label,
                "ATE": round(ate, 4),
                "SE": round(se, 4) if se is not None else np.nan,
                "n_controls": len(controls),
                "controls": ", ".join(controls) if controls else "(none)",
            }
        )
    return pd.DataFrame(rows)


def disparate_impact_ratio(
    data: pd.DataFrame, group_col: str, outcome: str, favorable_value=1
) -> float:
    """DIR: ratio of favorable-outcome rates between groups.

    For binary outcomes (0/1):
        DIR = P(Y=favorable | group=1) / P(Y=favorable | group=0)
    For continuous outcomes (e.g. COMPAS decile score, continuous Loan score):
        DIR = mean(Y | group=1) / mean(Y | group=0)

    Following the paper's convention. A DIR > 1.25 (or < 0.80, the 4/5 rule
    threshold) flags a disparity.
    """
    series = data[outcome].dropna()
    is_binary = series.nunique() <= 2 and set(series.unique()).issubset({0, 1, 0.0, 1.0})
    if is_binary:
        rate_minority = (data.loc[data[group_col] == 1, outcome] == favorable_value).mean()
        rate_majority = (data.loc[data[group_col] == 0, outcome] == favorable_value).mean()
        if rate_majority == 0:
            return float("inf")
        return float(rate_minority / rate_majority)
    else:
        mean_minority = data.loc[data[group_col] == 1, outcome].mean()
        mean_majority = data.loc[data[group_col] == 0, outcome].mean()
        if mean_majority == 0:
            return float("inf")
        return float(mean_minority / mean_majority)


def statistical_parity(
    data: pd.DataFrame, group_col: str, outcome: str
) -> float:
    """SP = P(Y | group=1) - P(Y | group=0).  0 = parity."""
    return float(
        data.loc[data[group_col] == 1, outcome].mean()
        - data.loc[data[group_col] == 0, outcome].mean()
    )


if __name__ == "__main__":
    from synthetic_data import generate_paired_datasets

    A, B = generate_paired_datasets(n=5000, beta_biased=-0.15, seed=42)

    stages = [
        ("no controls", []),
        ("+ Income only", ["Income"]),
        ("+ ZIP, CreditSc, Income", ["ZIP", "CreditSc", "Income"]),
    ]

    print("Biased dataset (true beta_Race->Loan = -0.15 in the latent logit):")
    print(staged_backdoor_ate(A, "Race", "Loan", stages))
    print()
    print("Unbiased dataset (true beta_Race->Loan = 0):")
    print(staged_backdoor_ate(B, "Race", "Loan", stages))
    print()
    print(f"DIR (biased)  : {disparate_impact_ratio(A, 'Race', 'Loan'):.3f}")
    print(f"DIR (unbiased): {disparate_impact_ratio(B, 'Race', 'Loan'):.3f}")
