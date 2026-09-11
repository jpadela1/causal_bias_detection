""""
synthetic_data.py
=================
Generate the synthetic loan-approval datasets used in Study 1 of the paper.

Structural Causal Model (SCM)
-----------------------------
Latent:
    SES ~ N(0, 1)                                            (hidden confounder
                                                                 of Education & Income;
                                                                 detectable by FCI as a
                                                                 bidirected edge)

Protected attributes:
    Race    ~ Bernoulli(0.5)        (1 = minority)
    Gender  ~ Bernoulli(0.5)        (1 = male)

Covariates:
    Education = 0.40*SES + 0.30*Gender + e_E,      e_E ~ Uniform(-1, 1)
    ZIP       = -0.50*Race + e_Z,                   e_Z ~ Uniform(-1, 1)
    Income    = 0.30*Education + 0.40*SES
                - 0.20*Race + 0.15*Gender + e_I,    e_I ~ Uniform(-1, 1)
    CreditSc  = 0.40*Education + 0.30*ZIP
                + 0.30*Income + e_C,                e_C ~ Uniform(-1, 1)

Outcome (continuous "loan-approval score" by default):
    Loan = 0.5*CreditSc + 0.4*Income + beta*Race + e_L,    e_L ~ Uniform(-1, 1)


Two discrimination channels are planted in the biased dataset:
    * DIRECT : Race -> Loan                       (coefficient beta = -0.15)
    * PROXY  : Race -> ZIP -> CreditSc -> Loan     (-0.50 * 0.30 * 0.50)
      ZIP is a facially-neutral proxy: it carries Race's influence to the
      outcome through CreditSc without a direct Race->Loan edge.

Datasets:
    Dataset A (biased)   : beta = -0.15  (planted direct discrimination)
    Dataset B (unbiased) : beta =  0.00

Non-Gaussian (Uniform) noise is required for LiNGAM identifiability.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


VARIABLE_ORDER = ["Race", "Gender", "Education", "ZIP", "Income", "CreditSc", "Loan"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_loan_data(
    n: int = 5000,
    beta: float = -0.15,
    seed: int = 42,
    return_latent: bool = False,
    binary_outcome: bool = False,
) -> pd.DataFrame:
    """Generate one synthetic loan-approval dataset.

    Parameters
    ----------
    n : int
        Sample size.
    beta : float
        Coefficient on the planted Race -> Loan edge. Use 0.0 for unbiased.
    seed : int
        RNG seed (42).
    return_latent : bool
        If True, include the hidden SES column (for diagnostics only).
    binary_outcome : bool
        If True, pass the continuous Loan score through a sigmoid + Bernoulli
        draw to mimic an actual approve/deny decision. Default False (linear)
        because LiNGAM's identification result requires a linear outcome.

    Returns
    -------
    pd.DataFrame with columns in VARIABLE_ORDER
    (and 'SES' appended if return_latent=True).
    """
    rng = np.random.default_rng(seed)

    # Latent confounder (NOT returned unless return_latent=True)
    SES = rng.standard_normal(n)

    # Protected attributes
    Race = rng.binomial(1, 0.5, size=n).astype(float)
    Gender = rng.binomial(1, 0.5, size=n).astype(float)

    # Non-Gaussian noise (required for LiNGAM identifiability)
    eE = rng.uniform(-1.0, 1.0, size=n)
    eZ = rng.uniform(-1.0, 1.0, size=n)
    eI = rng.uniform(-1.0, 1.0, size=n)
    eC = rng.uniform(-1.0, 1.0, size=n)
    eL = rng.uniform(-1.0, 1.0, size=n)

    Education = 0.40 * SES + 0.30 * Gender + eE
    ZIP = -0.50 * Race + eZ
    Income = (
        0.30 * Education
        + 0.40 * SES
        - 0.20 * Race
        + 0.15 * Gender
        + eI
    )
    CreditSc = 0.40 * Education + 0.30 * ZIP + 0.30 * Income + eC

    # Linear outcome (LiNGAM-coefficient-recovery setup)
    Loan_score = 0.5 * CreditSc + 0.4 * Income + beta * Race + eL

    if binary_outcome:
        Loan = rng.binomial(1, _sigmoid(Loan_score)).astype(float)
    else:
        Loan = Loan_score

    data = {
        "Race": Race,
        "Gender": Gender,
        "Education": Education,
        "ZIP": ZIP,
        "Income": Income,
        "CreditSc": CreditSc,
        "Loan": Loan,
    }
    if return_latent:
        data["SES"] = SES

    return pd.DataFrame(data)


def generate_paired_datasets(
    n: int = 5000,
    beta_biased: float = -0.15,
    seed: int = 42,
    binary_outcome: bool = False,
):
    """Convenience wrapper returning (Dataset A biased, Dataset B unbiased)."""
    biased = generate_loan_data(n=n, beta=beta_biased, seed=seed,
                                binary_outcome=binary_outcome)
    unbiased = generate_loan_data(n=n, beta=0.0, seed=seed,
                                  binary_outcome=binary_outcome)
    return biased, unbiased


# --------------------------------------------------------------------------- #
# Ground-truth DAG specification
# --------------------------------------------------------------------------- #
# These coefficients mirror the SCM equations above 1:1. They drive both the
# data generation (already done in generate_loan_data) and the ground-truth
# DAG figure (plot_ground_truth_dag below). If you change one, change both.

GROUND_TRUTH_EDGES = [
    # (source, target, coefficient_label)
    ("SES",       "Education", "0.40"),     # latent confounder positive magnitude to the likelihood; higher ed->higher likelihood
    ("SES",       "Income",    "0.40"),     # latent confounder positive magnitude to the likelihood
    ("Gender",    "Education", "0.30"),
    ("Gender",    "Income",    "0.15"),
    ("Race",      "ZIP",       "-0.50"),
    ("Race",      "Income",    "-0.20"),
    ("Education", "Income",    "0.30"),
    ("Education", "CreditSc",  "0.40"),
    ("ZIP",       "CreditSc",  "0.30"),
    ("Income",    "CreditSc",  "0.30"),
    ("Income",    "Loan",      "0.40"),
    ("CreditSc",  "Loan",      "0.50"),
    ("Race",      "Loan",      "\u03b2=\u22120.15"),  # planted bias edge
]

# Node roles: drives node colors. Same scheme used in visualization.py.
GROUND_TRUTH_NODE_ROLES = {
    "SES":       "latent",
    "Race":      "protected",
    "Gender":    "protected",
    "ZIP":       "proxy",
    "Education": "covariate",
    "Income":    "mediator",
    "CreditSc":  "mediator",
    "Loan":      "outcome",
}

# Proxy-discrimination pathway: the two legs that make ZIP a proxy for Race.
# Race -> ZIP -> CreditSc carries Race's influence to the outcome without a
# direct Race->Loan edge (the remaining CreditSc -> Loan leg is shared with
# legitimate paths, so it is not highlighted as proxy-specific). These edges
# are drawn in the proxy color so the pathway reads as clearly as the planted
# direct-bias edge. Edit this set if the proxy structure changes.
PROXY_PATHWAY_EDGES = {("Race", "ZIP"), ("ZIP", "CreditSc")}


def plot_ground_truth_dag(
    save_path: str = "figures/ground_truth_dag",
    show_coefficients: bool = True,
    title: str = "Ground-Truth DAG: Synthetic Loan-Approval SCM",
    show_both_versions: bool = False,  #False shows the biased version only, True shows both versions
):
    """Render the SCM as a DAG figure, via the shared Graphviz renderer.

    Rendering is delegated to ``visualization.plot_edge_list`` — the same code
    path, and therefore the same node shapes, fills and legend, that every
    discovered DAG goes through. This module keeps only the SCM
    *specification* (GROUND_TRUTH_EDGES / GROUND_TRUTH_NODE_ROLES /
    PROXY_PATHWAY_EDGES), so the figure stays in sync with the equations
    above and cannot drift from the discovery figures' styling.

    The two discrimination channels stay visually distinct: the planted DIRECT
    bias (Race -> Loan) is red and dashed, and the PROXY pathway
    (Race -> ZIP -> CreditSc) is drawn in the proxy colour with a heavier
    stroke. Both cues survive greyscale printing, as does the role-to-shape
    encoding (SES, unobserved, is a dashed diamond).

    Parameters
    ----------
    save_path : str
        Basename (extension is stripped). Figure goes to <basename>.pdf/.png.
    show_coefficients : bool
        If True, label each edge with its structural coefficient.
    title : str
        Figure-level title.
    show_both_versions : bool
        If True, writes TWO figures -- "<save_path>_biased" (with Race -> Loan)
        and "<save_path>_unbiased" (without) -- and returns both. Graphviz lays
        out one graph per drawing, so the two SCMs are separate files rather
        than the side-by-side panels the old matplotlib version produced. If
        False, only the biased version is written, to <save_path>.

    Returns the ``graphviz.Digraph``, or a list of two when
    ``show_both_versions`` is True.
    """
    from visualization import plot_edge_list

    edges_biased   = list(GROUND_TRUTH_EDGES)
    edges_unbiased = [e for e in GROUND_TRUTH_EDGES
                      if not (e[0] == "Race" and e[1] == "Loan")]

    def _render(edges, path, panel_title):
        return plot_edge_list(
            edges,
            title             = panel_title,
            node_roles        = GROUND_TRUTH_NODE_ROLES,
            flagged_edges     = [("Race", "Loan")],
            proxy_edges       = PROXY_PATHWAY_EDGES,
            show_coefficients = show_coefficients,
            save_path         = path,
            flagged_label     = "Planted direct bias (Race \u2192 Loan)",
        )

    biased_title = f"{title}\nBiased SCM (Dataset A, \u03b2 = \u22120.15)"

    if not show_both_versions:
        return _render(edges_biased, save_path, biased_title)

    base = os.path.splitext(save_path)[0]
    return [
        _render(edges_biased, f"{base}_biased", biased_title),
        _render(edges_unbiased, f"{base}_unbiased",
                f"{title}\nUnbiased SCM (Dataset B, \u03b2 = 0.00)"),
    ]


if __name__ == "__main__":
    A, B = generate_paired_datasets()
    print("Dataset A (biased) head:")
    print(A.head())
    print("\nDataset A summary:")
    print(A.describe().round(3))
    print("\nMean Loan score by Race (biased):")
    print(A.groupby("Race")["Loan"].mean().round(3))
    print("\nMean Loan score by Race (unbiased):")
    print(B.groupby("Race")["Loan"].mean().round(3))