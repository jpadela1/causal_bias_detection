"""
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


Reset the n=5,000 including synthetic data samples, loan dataset

Datasets:
    Dataset A (biased)   : beta = -0.15  (planted direct discrimination)
    Dataset B (unbiased) : beta =  0.00

Non-Gaussian (Uniform) noise is required for LiNGAM identifiability.
"""
from __future__ import annotations

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


def plot_ground_truth_dag(
    save_path: str = "figures/ground_truth_dag",
    show_coefficients: bool = True,
    title: str = "Ground-Truth DAG: Synthetic Loan-Approval SCM",
    show_both_versions: bool = True,
):
    """Render the SCM as a DAG figure.

    Parameters
    ----------
    save_path : str
        Basename (extension is stripped). Figure goes to <basename>.png/pdf.
    show_coefficients : bool
        If True, label each edge with its structural coefficient.
    title : str
        Figure-level title.
    show_both_versions : bool
        If True, draws two side-by-side panels: BIASED (with Race -> Loan)
        and UNBIASED (without). If False, draws only the biased version.
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Patch

    # Layered LEFT-TO-RIGHT positions

    pos = {
        # column 0 (left): exogenous protected attributes + latent confounder
        "Race":      (-3.0,  1.8),
        "Gender":    (-3.0,  0.0),
        "SES":       (-3.0, -1.8),
        # column 1: ZIP (Race-driven) and Education (Gender + SES driven)
        "ZIP":       (-1.0,  1.8),
        "Education": (-1.0, -0.5),
        # column 2: Income (mediator with many parents)
        "Income":    ( 1.0,  0.6),
        # column 3: CreditSc
        "CreditSc":  ( 3.0,  0.6),
        # column 4 (right): Loan outcome, sink
        "Loan":      ( 5.0,  0.6),
    }

    role_colors = {
        "latent":    "#FFFFFF",   # white fill, dashed border (drawn below)
        "protected": "#A6CEE3",   # soft sky blue
        "proxy":     "#FDB863",   # pale orange
        "covariate": "#CCCCCC",   # light gray
        "mediator":  "#A6DBA0",   # pastel green
        "outcome":   "#FBB4AE",   # soft pink
    }
    role_labels = {
        "latent":    "Latent confounder (unobserved)",
        "protected": "Protected attribute",
        "proxy":     "Proxy variable",
        "covariate": "Covariate",
        "mediator":  "Mediator",
        "outcome":   "Outcome",
    }

    NODE_SIZE = 1700
    NODE_RADIUS_PT = math.sqrt(NODE_SIZE / math.pi)
    EXTRA_MARGIN = 4.0

    def _draw_panel(ax, edges_to_draw, panel_title: str):
        # Edges first
        for src, dst, coef in edges_to_draw:
            is_planted_bias = (src == "Race" and dst == "Loan")
            is_latent_edge = (src == "SES")
            color = "#D62728" if is_planted_bias else (
                "#888888" if is_latent_edge else "#333333"
            )
            style = "dashed" if (is_planted_bias or is_latent_edge) else "solid"
            lw = 2.4 if is_planted_bias else (1.2 if is_latent_edge else 1.6)
            # Curve the planted-bias edge strongly so it sweeps along
            # the BOTTOM of the layout past the lower nodes instead of
            # cutting through the middle. Positive rad = curve below
            # the line connecting source to target.
            if is_planted_bias:
                rad = 0.45
            elif is_latent_edge:
                rad = 0.05
            else:
                rad = 0.05
            arrow = FancyArrowPatch(
                posA=pos[src], posB=pos[dst],
                arrowstyle="-|>", mutation_scale=22 if is_planted_bias else 20,
                color=color, linewidth=lw, linestyle=style,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=NODE_RADIUS_PT + EXTRA_MARGIN,
                shrinkB=NODE_RADIUS_PT + EXTRA_MARGIN,
                zorder=3 if is_planted_bias else 2,
            )
            ax.add_patch(arrow)

            if show_coefficients:
                MANUAL_LABEL_POS = {
                    # SES (-3, -1.8) -> Education (-1, -0.5)
                    ("SES",       "Education"): (-2.4, -1.3),
                    # SES (-3, -1.8) -> Income (1, 0.6)
                    ("SES",       "Income"):    ( 0.0, -1.4),
                    # Gender (-3, 0) -> Education (-1, -0.5)
                    ("Gender",    "Education"): (-2.4, -0.4),
                    # Gender (-3, 0) -> Income (1, 0.6)
                    ("Gender",    "Income"):    (-1.5,  0.6),
                    # Race (-3, 1.8) -> ZIP (-1, 1.8)
                    ("Race",      "ZIP"):       (-2.0,  2.1),
                    # Race (-3, 1.8) -> Income (1, 0.6)
                    ("Race",      "Income"):    (-0.7,  1.6),
                    # Race -> Loan (curved bottom): apex around (1, -0.6)
                    ("Race",      "Loan"):      ( 1.0, -1.1),
                    # ZIP (-1, 1.8) -> CreditSc (3, 0.6)
                    ("ZIP",       "CreditSc"):  ( 1.0,  1.5),
                    # Education (-1, -0.5) -> Income (1, 0.6)
                    ("Education", "Income"):    (-0.3,  0.4),
                    # Education (-1, -0.5) -> CreditSc (3, 0.6)
                    ("Education", "CreditSc"):  ( 1.5, -0.2),
                    # Income (1, 0.6) -> CreditSc (3, 0.6)
                    ("Income",    "CreditSc"):  ( 2.0,  0.95),
                    # Income (1, 0.6) -> Loan (5, 0.6) -- top arc
                    ("Income",    "Loan"):      ( 3.0,  1.05),
                    # CreditSc (3, 0.6) -> Loan (5, 0.6)
                    ("CreditSc",  "Loan"):      ( 4.0,  0.95),
                }
                if (src, dst) in MANUAL_LABEL_POS:
                    lab_x, lab_y = MANUAL_LABEL_POS[(src, dst)]
                else:
                    # Algorithmic fallback: 45% along the edge with a
                    # perpendicular nudge.
                    dx = pos[dst][0] - pos[src][0]
                    dy = pos[dst][1] - pos[src][1]
                    edge_len = (dx * dx + dy * dy) ** 0.5 or 1.0
                    t = 0.45 if edge_len < 2.0 else 0.55
                    base_x = pos[src][0] + t * dx
                    base_y = pos[src][1] + t * dy
                    perp_x = -dy / edge_len
                    perp_y = dx / edge_len
                    offset = 0.22 if base_x >= 0 else -0.22
                    lab_x = base_x + perp_x * offset
                    lab_y = base_y + perp_y * offset

                label_color = ("#D62728" if is_planted_bias else
                               ("#666666" if is_latent_edge else "#222222"))
                ax.text(
                    lab_x, lab_y, coef, fontsize=8, ha="center", va="center",
                    color=label_color, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="#dddddd",
                              alpha=0.95, boxstyle="round,pad=0.18",
                              linewidth=0.5),
                    zorder=6,
                )

        # Nodes
        for v, (x, y) in pos.items():
            role = GROUND_TRUTH_NODE_ROLES.get(v, "covariate")
            face = role_colors[role]
            edge_style = "--" if role == "latent" else "-"
            edge_lw = 2.0 if role == "latent" else 1.5
            edge_color = "#666666" if role == "latent" else "black"
            ax.scatter(
                x, y, s=NODE_SIZE, c=face, edgecolors=edge_color,
                linewidths=edge_lw, zorder=4,
                 )
            if role == "latent":
                # Overlay a dashed circle for the latent node
                circle = plt.Circle(
                    (x, y), 0.155, fill=False,
                    edgecolor=edge_color, linewidth=edge_lw,
                    linestyle=edge_style, zorder=4.5,
                )
                ax.add_patch(circle)
            ax.annotate(v, xy=(x, y), ha="center", va="center",
                        fontsize=9, fontweight="bold", zorder=5)

        ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(-4.0, 6.0)
        ax.set_ylim(-3.2, 2.8)
        ax.set_aspect("equal")
        ax.set_axis_off()

    # --- Figure assembly ----------------------------------------------------
    if show_both_versions:
        fig, (ax_b, ax_u) = plt.subplots(1, 2, figsize=(18, 6.5))
        edges_biased = GROUND_TRUTH_EDGES
        edges_unbiased = [e for e in GROUND_TRUTH_EDGES
                          if not (e[0] == "Race" and e[1] == "Loan")]
        _draw_panel(ax_b, edges_biased,
                    "Biased SCM (Dataset A, β = −0.15)")
        _draw_panel(ax_u, edges_unbiased,
                    "Unbiased SCM (Dataset B, β = 0.00)")
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        _draw_panel(ax, GROUND_TRUTH_EDGES,
                    "Biased SCM (Dataset A, β = −0.15)")

    # --- Legend ------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], color="#333333", lw=1.6, marker=">",
               markersize=9, label="Observed causal edge"),
        Line2D([0], [0], color="#888888", lw=1.2, linestyle="--",
               marker=">", markersize=9, label="Edge from latent SES"),
        Line2D([0], [0], color="#D62728", lw=2.4, linestyle="--",
               marker=">", markersize=9, label="Planted direct bias (Race → Loan)"),
    ]
    legend_handles.extend([
        Patch(facecolor=role_colors[role], edgecolor="black",
              label=role_labels[role])
        for role in ["protected", "proxy", "covariate", "mediator",
                     "outcome", "latent"]
    ])

    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.03, right=0.97)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=True, fancybox=True, framealpha=0.95,
               edgecolor="#cccccc", fontsize=9,
               bbox_to_anchor=(0.5, 0.02))

    # Save both formats
    from visualization import save_figure_dual_format
    save_figure_dual_format(fig, save_path)
    plt.close(fig)
    return fig


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
