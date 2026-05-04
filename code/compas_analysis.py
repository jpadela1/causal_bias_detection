"""
compas_analysis.py
==================
Load and preprocess the COMPAS dataset (Section VI of the paper) and run the
full causal-discovery + ATE pipeline against it.

Source CSV
----------
ProPublica's compas-scores.csv:
    https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores.csv

If you cannot download from the script (offline / firewall), drop the file at
    data/compas-scores.csv
and the loader will pick it up.

ProPublica preprocessing filters
--------------------------------
* |days_b_screening_arrest| <= 30
* is_recid != -1
* c_charge_degree in {"F", "M"}
* score_text != "N/A"

Encoded variables (matching the paper's nine-variable schema):
    Race          : 1 = African-American, 0 = other
    Sex           : 1 = Male, 0 = Female
    Age           : age in years
    JuvFelony     : juv_fel_count
    JuvMisd       : juv_misd_count
    Priors        : priors_count
    ChargeDegree  : 1 = Felony (F), 0 = Misdemeanor (M)
    Score         : decile_score (1-10)
    Recidivism    : two_year_recid (0 or 1)   # is_recid is sometimes used too
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

PROPUBLICA_CSV_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores.csv"
)

LOCAL_CSV_PATHS = [
    "data/compas-scores.csv",
    "compas-scores.csv",
]


COMPAS_COLUMNS = [
    "Race",
    "Sex",
    "Age",
    "JuvFelony",
    "JuvMisd",
    "Priors",
    "ChargeDegree",
    "Score",
    "Recidivism",
]


def _try_local_csv() -> Optional[pd.DataFrame]:
    for p in LOCAL_CSV_PATHS:
        if os.path.exists(p):
            print(f"Loading COMPAS from local file: {p}")
            return pd.read_csv(p)
    return None


def load_compas(use_local_only: bool = False) -> pd.DataFrame:
    """Load the raw ProPublica CSV (no preprocessing applied yet)."""
    raw = _try_local_csv()
    if raw is not None:
        return raw
    if use_local_only:
        raise FileNotFoundError(
            f"No local COMPAS CSV found in {LOCAL_CSV_PATHS}. "
            f"Download from {PROPUBLICA_CSV_URL} and place it under data/."
        )
    print(f"Downloading COMPAS from {PROPUBLICA_CSV_URL} ...")
    raw = pd.read_csv(PROPUBLICA_CSV_URL)
    os.makedirs("data", exist_ok=True)
    raw.to_csv("data/compas-scores.csv", index=False)
    print("  cached at data/compas-scores.csv")
    return raw


def preprocess_compas(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply ProPublica filters and encode the nine analysis variables."""
    df = raw.copy()

    # Standard ProPublica filters
    if "days_b_screening_arrest" in df.columns:
        df = df[df["days_b_screening_arrest"].abs() <= 30]
    if "is_recid" in df.columns:
        df = df[df["is_recid"] != -1]
    if "c_charge_degree" in df.columns:
        df = df[df["c_charge_degree"].isin(["F", "M"])]
    if "score_text" in df.columns:
        df = df[df["score_text"].notna() & (df["score_text"] != "N/A")]

    # Encode
    out = pd.DataFrame()
    out["Race"] = (df["race"] == "African-American").astype(int)
    out["Sex"] = (df["sex"] == "Male").astype(int)
    out["Age"] = df["age"].astype(float)
    out["JuvFelony"] = df["juv_fel_count"].astype(float)
    out["JuvMisd"] = df["juv_misd_count"].astype(float)
    out["Priors"] = df["priors_count"].astype(float)
    out["ChargeDegree"] = (df["c_charge_degree"] == "F").astype(int)
    out["Score"] = df["decile_score"].astype(float)
    # The paper uses two_year_recid as the actual outcome; fall back to is_recid.
    if "two_year_recid" in df.columns:
        out["Recidivism"] = df["two_year_recid"].astype(int)
    else:
        out["Recidivism"] = df["is_recid"].astype(int)

    out = out.dropna().reset_index(drop=True)
    return out[COMPAS_COLUMNS]


def baseline_disparities(df: pd.DataFrame) -> dict:
    """Reproduce the paper's baseline numbers: mean Score by race, DIR, parity."""
    score_minority = df.loc[df["Race"] == 1, "Score"].mean()
    score_majority = df.loc[df["Race"] == 0, "Score"].mean()
    rec_minority = df.loc[df["Race"] == 1, "Recidivism"].mean()
    rec_majority = df.loc[df["Race"] == 0, "Recidivism"].mean()

    return {
        "mean_score_AA": round(score_minority, 3),
        "mean_score_other": round(score_majority, 3),
        "DIR_score": round(score_minority / score_majority, 3) if score_majority else float("inf"),
        "DIR_recidivism": round(rec_minority / rec_majority, 3) if rec_majority else float("inf"),
        "stat_parity_score": round(score_minority - score_majority, 3),
        "stat_parity_recidivism": round(rec_minority - rec_majority, 3),
        "n_total": len(df),
        "n_AA": int((df["Race"] == 1).sum()),
        "n_other": int((df["Race"] == 0).sum()),
    }


# --------------------------------------------------------------------------- #
# COMPAS Hypothesized Ground-Truth DAG
# --------------------------------------------------------------------------- #
# Unlike the synthetic dataset, the TRUE causal graph for COMPAS is unknown.
# What we CAN write down is a "hypothesized" DAG that encodes minimal
# uncontroversial domain assumptions: the temporal/biological order in which
# variables come into existence, and which edges are forbidden by that order.
# Reviewers will accept SHD against this hypothesized DAG as long as it's
# clearly labeled "hypothesized" and the assumptions are stated.
#
# Assumptions (each citable to ProPublica 2016 or basic temporal logic):
#
# 1. EXOGENOUS (no incoming edges):
#    - Race, Sex   : determined at birth, immutable
#    - Age         : a temporal coordinate, not caused by the other variables
#                    in this dataset (NB: Race -> Age in some studies, but at
#                    the time of arrest Age is fixed and exogenous to the
#                    variables we observe)
#
# 2. CRIMINAL HISTORY (caused by demographics, accumulates over time):
#    - JuvFelony, JuvMisd    : juvenile-court records, by adulthood are fixed
#    - Priors                : adult criminal history accumulates with Age
#    - ChargeDegree          : characteristics of the CURRENT charge
#
#    Demographics influence these via well-documented socioeconomic and
#    enforcement-disparity pathways (e.g., Alexander 2010, ProPublica 2016).
#
# 3. SCORE (determined by the COMPAS algorithm at arraignment):
#    - Score is a function of demographics, criminal history, and charge.
#    - Whether Score directly depends on Race is the empirical question
#      under audit -- so we INCLUDE it as a hypothesized edge in the
#      "biased" reference but mark it "auditable" rather than confirmed.
#
# 4. RECIDIVISM (the future outcome the score tries to predict):
#    - Recidivism is the latest event in time -- everything else precedes it.
#    - Score should NOT directly cause recidivism (the score is a prediction
#      tool, not a treatment), but Score may be correlated with recidivism
#      via shared causes. We therefore do NOT include Score -> Recidivism.
#
# Edge list below uses (source, target) format and can be passed to
# structural_hamming_distance() from causal_discovery.py.

COMPAS_GROUND_TRUTH_EDGES = [
    # Demographics -> criminal history pathways
    ("Age",          "JuvFelony"),
    ("Age",          "JuvMisd"),
    ("Age",          "Priors"),
    ("Race",         "Priors"),       # well-documented enforcement disparity
    ("Sex",          "Priors"),       # base-rate difference by sex
    ("Sex",          "ChargeDegree"),

    # Juvenile -> adult history
    ("JuvFelony",    "Priors"),
    ("JuvMisd",      "Priors"),

    # History -> COMPAS score (legitimate criminogenic features)
    ("Priors",       "Score"),
    ("ChargeDegree", "Score"),
    ("Age",          "Score"),

    # The auditable bias edge -- present in the "biased" hypothesis,
    # absent in the "fair" hypothesis. SHD should be computed against
    # whichever the user is asking about.
    ("Race",         "Score"),

    # History -> recidivism (criminogenic features predict reoffending)
    ("Priors",       "Recidivism"),
    ("ChargeDegree", "Recidivism"),
    ("Age",          "Recidivism"),
    ("Sex",          "Recidivism"),
]

# Without the auditable Race -> Score edge -- a "fair" reference structure.
# Compute SHD against this when asking "did the algorithm correctly OMIT
# the discrimination edge?"
COMPAS_GROUND_TRUTH_EDGES_FAIR = [
    e for e in COMPAS_GROUND_TRUTH_EDGES
    if not (e[0] == "Race" and e[1] == "Score")
]


def get_compas_ground_truth(reference: str = "biased") -> list:
    """Return the hypothesized COMPAS ground-truth edge list.

    Parameters
    ----------
    reference : 'biased' or 'fair'
        - 'biased': includes the auditable Race -> Score edge. Use when you
          want SHD to count "missing the discrimination edge" as a structural
          error (i.e., the literature's null hypothesis is that COMPAS IS
          biased, and an algorithm that doesn't surface this is wrong).
        - 'fair': excludes Race -> Score. Use when you want SHD to count
          "spuriously detecting Race -> Score" as a false positive.

    Reporting both is best. The paper figure can then say "SHD vs biased
    reference: X" and "SHD vs fair reference: Y", showing the algorithm's
    cost under each hypothesis.
    """
    if reference == "biased":
        return list(COMPAS_GROUND_TRUTH_EDGES)
    if reference == "fair":
        return list(COMPAS_GROUND_TRUTH_EDGES_FAIR)
    raise ValueError(f"reference must be 'biased' or 'fair', got {reference!r}")


def plot_correlation_vs_causal(
    correlation_metrics: dict,
    causal_estimates: dict,
    save_path: str = "figures/compas_corr_vs_causal",
    title: str = "Correlation-Based vs. Causal Methods for COMPAS Bias Detection",
    subtitle: str = (
        "Causal methods isolate the direct Race \u2192 Score pathway "
        "from legitimate criminal-history pathways"
    ),
):
    """Two-panel comparison: correlation-based metrics vs causal estimates.

    Reproduces the paper's "Correlation-Based vs Causal" figure but builds
    every bar from values you pass in, so it always reflects your live run.

    Parameters
    ----------
    correlation_metrics : dict
        Mapping label -> value for the LEFT panel. Typical keys (any subset
        is fine; only entries you provide are plotted)::

            {
                "COMPAS Score\nDisparate Impact\n(\u22650.8 = fair)": 1.545,
                "COMPAS Score\nStat. Parity\n(=0 = fair)":          1.545,
                "Recidivism\nDisparate Impact":                     1.471,
            }

    causal_estimates : dict
        Mapping label -> value for the RIGHT panel. To handle the case where
        ICA-LiNGAM and DirectLiNGAM disagree, pass BOTH as separate entries::

            {
                "ATE Race\u2192Score\n(no controls)":   0.324,
                "ATE Race\u2192Score\n(full controls)": 0.138,
                "DirectLiNGAM \u03b2\nRace\u2192Score": 0.134,
                "ICA-LiNGAM \u03b2\nRace\u2192Score":   0.091,
            }

        The function automatically draws a delta-arrow between the two
        adjacent bars whose absolute values are most similar (typically
        full-controls ATE vs the closest LiNGAM \u03b2). To suppress that
        arrow, pass ``causal_estimates`` with only one ATE entry.

    save_path : str
        Basename. Saves both ``.png`` and ``.pdf`` via ``save_figure_dual_format``.
    """
    import matplotlib.pyplot as plt
    from visualization import save_figure_dual_format

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))

    # ===== LEFT panel: correlation-based metrics ============================
    corr_labels = list(correlation_metrics.keys())
    corr_values = list(correlation_metrics.values())
    bar_color_corr = "#C0392B"   # red

    axL.bar(range(len(corr_labels)), corr_values,
            color=bar_color_corr, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(corr_values):
        axL.text(i, v + 0.04, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=bar_color_corr)

    # 4/5 rule reference line at y=0.8
    axL.axhline(0.8, color="black", linestyle="--", linewidth=1.5,
                label="4/5 rule threshold (0.8)")
    # Faint reference at parity (1.0 for DIR, 0 for stat. parity)
    axL.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    axL.set_xticks(range(len(corr_labels)))
    axL.set_xticklabels(corr_labels, fontsize=9)
    axL.set_ylabel("Metric value", fontsize=11)
    axL.set_ylim(0, max(2.0, max(corr_values) * 1.2 if corr_values else 2.0))
    axL.set_title(
        "Correlation-Based Fairness Metrics\n(Detect THAT bias exists)",
        fontsize=12, fontweight="bold",
    )
    axL.legend(loc="upper right", fontsize=10, frameon=True)
    axL.grid(axis="y", linestyle=":", alpha=0.3)

    # ===== RIGHT panel: causal estimates =====================================
    causal_labels = list(causal_estimates.keys())
    causal_values = list(causal_estimates.values())

    # Light-to-dark grey gradient: gives "from less adjusted -> more adjusted"
    # a visual narrative without making bars feel competitive.
    n_causal = len(causal_values)
    if n_causal == 0:
        greys = []
    elif n_causal == 1:
        greys = ["#5A5A5A"]
    else:
        # Interpolate from light grey (#C8C8C8) to near-black (#3C3C3C)
        greys = []
        for i in range(n_causal):
            level = int(200 - 140 * i / max(n_causal - 1, 1))
            greys.append(f"#{level:02x}{level:02x}{level:02x}")

    axR.bar(range(n_causal), causal_values, color=greys,
            edgecolor="black", linewidth=0.5)
    for i, v in enumerate(causal_values):
        axR.text(i, v + 0.008, f"{v:+.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    # Auto-detect a "controlled-vs-LiNGAM agreement" pair to annotate. We
    # look for the most-similar adjacent pair (smallest |delta|) among
    # everything except the very first bar (which is the unadjusted ATE
    # and would dominate). If no good candidate, skip the arrow silently.
    if n_causal >= 3:
        best_pair = None
        best_delta = float("inf")
        for i in range(1, n_causal - 1):
            d = abs(causal_values[i] - causal_values[i + 1])
            if d < best_delta:
                best_delta = d
                best_pair = (i, i + 1)
        # Only annotate if the two bars are visibly similar (within 25% of
        # the larger one). Otherwise drawing an arrow misleads.
        if best_pair is not None:
            v1, v2 = causal_values[best_pair[0]], causal_values[best_pair[1]]
            if max(abs(v1), abs(v2)) > 0 and best_delta / max(abs(v1), abs(v2)) < 0.25:
                y_arrow = max(v1, v2) * 0.55
                axR.annotate(
                    "", xy=(best_pair[1], y_arrow), xytext=(best_pair[0], y_arrow),
                    arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.8),
                )
                axR.text(
                    (best_pair[0] + best_pair[1]) / 2, y_arrow + 0.012,
                    f"\u0394={best_delta:.4f}", ha="center", va="bottom",
                    fontsize=9, color="#C0392B", fontweight="bold",
                )

    # Side annotation: explain the sign of the estimates
    axR.text(
        0.98, 0.97,
        "Positive \u03b2 = minority\ngroup over-scored\nindependent of risk",
        transform=axR.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDF2E0",
                  edgecolor="#D6A85F", linewidth=1.0),
    )

    axR.set_xticks(range(n_causal))
    axR.set_xticklabels(causal_labels, fontsize=9)
    axR.set_ylabel("Causal effect estimate (std. units)", fontsize=11)
    axR.set_ylim(0, max(0.4, max(causal_values) * 1.25 if causal_values else 0.4))
    axR.set_title(
        "Causal Estimates\n(Quantify WHY and HOW MUCH bias exists)",
        fontsize=12, fontweight="bold",
    )
    axR.grid(axis="y", linestyle=":", alpha=0.3)

    # ===== Figure-level title and layout ===================================
    # Reserve space at top: row order is suptitle, subtitle, panel titles.
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.97,
                        wspace=0.25)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)
    fig.text(0.5, 0.88, subtitle, ha="center", va="center", fontsize=10,
             style="italic", color="#444")
    save_figure_dual_format(fig, save_path)
    plt.close(fig)
    return fig



    df = preprocess_compas(raw)
    print(f"Preprocessed COMPAS: n={len(df)} rows, {df.shape[1]} columns")
    print(df.head())
    print()
    for k, v in baseline_disparities(df).items():
        print(f"  {k:25s} {v}")
