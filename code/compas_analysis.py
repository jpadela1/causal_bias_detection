"""
compas_analysis.py
==================
Load and preprocess the ProPublica COMPAS dataset for the paper's Section VI
analysis, and compute the baseline correlation-based fairness metrics.

Source CSV
----------
ProPublica's compas-scores-two-years.csv:
    https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv

This is the file with the dedicated `two_year_recid` column (730-day window),
which is what the literature reports rates against. The raw `compas-scores.csv`
in the same repo uses `is_recid` over a longer/variable observation window and
should NOT be used when reporting "two-year recidivism rates" -- that was the
mismatch in the paper's earlier draft.

If you cannot download (offline / firewall), drop the file at
    data/compas-scores-two-years.csv
and the loader will pick it up.

ProPublica preprocessing filters
--------------------------------
* |days_b_screening_arrest| <= 30
* is_recid != -1
* c_charge_degree in {"F", "M"}
* score_text != "N/A"

Encoded variables (matching the paper's nine-variable schema):
    Race          : 1 = African-American, 0 = Caucasian
    Sex           : 1 = Male,             0 = Female
    Age           : age in years
    JuvFelony     : juv_fel_count
    JuvMisd       : juv_misd_count
    Priors        : priors_count
    ChargeDegree  : 1 = Felony (F),       0 = Misdemeanor (M)
    Score         : decile_score (1-10)
    Recidivism    : two_year_recid (0/1)

Race policy
-----------
The paper's published comparison is "African-American vs Caucasian" (the same
two-group split ProPublica reports). By default `preprocess_compas` therefore
*filters* to just those two groups so the binary Race=1/0 encoding used by the
causal discovery algorithms is unambiguous and the reported disparities match
the literature. Pass `restrict_to_aa_caucasian=False` to keep all races (Race=0
will then mean "any non-African-American").
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

PROPUBLICA_CSV_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
    "compas-scores-two-years.csv"
)

LOCAL_CSV_PATHS = [
    "data/compas-scores-two-years.csv",
    "compas-scores-two-years.csv",
]

CACHE_PATH = "data/compas-scores-two-years.csv"


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

# Decile threshold used to define "high risk" for the proper selection-rate
# DIR. ProPublica's COMPAS labels Low = 1-4, Medium = 5-7, High = 8-10. The
# paper's adverse-classification analysis follows ProPublica in treating any
# score >= 5 (Medium or High) as the adverse classification.
HIGH_RISK_THRESHOLD = 5


def _try_local_csv() -> Optional[pd.DataFrame]:
    for p in LOCAL_CSV_PATHS:
        if os.path.exists(p):
            print(f"Loading COMPAS from local file: {p}")
            return pd.read_csv(p)
    return None


def load_compas(use_local_only: bool = False) -> pd.DataFrame:
    """Load the raw ProPublica two-year CSV (no preprocessing applied yet)."""
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
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    raw.to_csv(CACHE_PATH, index=False)
    print(f"  cached at {CACHE_PATH}")
    return raw


def preprocess_compas(
    raw: pd.DataFrame,
    restrict_to_aa_caucasian: bool = True,
) -> pd.DataFrame:
    """Apply ProPublica filters and encode the nine analysis variables.

    Parameters
    ----------
    restrict_to_aa_caucasian : bool, default True
        If True, drops all defendants who are not African-American or
        Caucasian. The resulting Race=1/0 encoding is then unambiguously
        "African-American vs Caucasian", matching ProPublica's headline
        comparison and the paper's text.
        If False, Race=1 is African-American and Race=0 is everyone else
        (Caucasian + Hispanic + Asian + Native American + Other).
    """
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

    if restrict_to_aa_caucasian:
        df = df[df["race"].isin(["African-American", "Caucasian"])]

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
    if "two_year_recid" in df.columns:
        out["Recidivism"] = df["two_year_recid"].astype(int)
    else:
        # Fallback for compas-scores.csv (the non-two-years file). This path
        # should not normally be hit -- we cache the two-years CSV by default.
        print("  WARNING: two_year_recid column not present; falling back to "
              "is_recid. Rates from this column do NOT correspond to a "
              "two-year window.")
        out["Recidivism"] = df["is_recid"].astype(int)

    out = out.dropna().reset_index(drop=True)
    return out[COMPAS_COLUMNS]


# --------------------------------------------------------------------------- #
# Baseline correlation-based disparities
# --------------------------------------------------------------------------- #

def _safe_ratio(num: float, den: float) -> float:
    """Return num/den, or float('inf') if den == 0."""
    if den == 0 or den is None or np.isnan(den):
        return float("inf")
    return num / den


def baseline_disparities(df: pd.DataFrame) -> dict:
    """Compute the paper's Section VI-A baseline numbers.

    All metrics are reported with explicit names so the figure / paper text
    cannot accidentally compare a mean ratio against the 4/5-rule threshold.

    Returns a dict with:

      Sample sizes
        n_total, n_AA, n_other

      Group means (descriptive)
        mean_score_AA, mean_score_other

      Recidivism (rates -- comparable to 0.80 threshold)
        recid_rate_AA, recid_rate_other
        DIR_recidivism                  P(recid|AA) / P(recid|other)

      Score disparity, the WRONG way (mean ratio -- NOT comparable to 0.80)
        mean_score_ratio                mean(Score|AA) / mean(Score|other)
        stat_parity_score               mean(Score|AA) - mean(Score|other)

      Score disparity, the RIGHT way (selection-rate DIR -- comparable to 0.80)
        high_risk_rate_AA               P(Score >= 5 | AA)
        high_risk_rate_other            P(Score >= 5 | other)
        DIR_score_selection_rate        P(Score>=5|AA) / P(Score>=5|other)

    Notes
    -----
    The "selection rate" DIR is what disparate-impact case law and the EEOC
    four-fifths rule are actually defined for: a ratio of binary outcome
    rates, bounded in [0, 1] when the favored group is in the denominator.
    Comparing a mean-of-ordinal-scores ratio against 0.80 -- as the paper's
    earlier draft did -- is dimensionally incorrect, and reviewers will flag
    it. Both numbers are returned so the figure can show them side by side
    and the text can comment on the contrast.
    """
    minority = df[df["Race"] == 1]
    majority = df[df["Race"] == 0]

    score_AA  = float(minority["Score"].mean())
    score_oth = float(majority["Score"].mean())

    rec_AA  = float(minority["Recidivism"].mean())
    rec_oth = float(majority["Recidivism"].mean())

    high_AA  = float((minority["Score"] >= HIGH_RISK_THRESHOLD).mean())
    high_oth = float((majority["Score"] >= HIGH_RISK_THRESHOLD).mean())

    return {
        # Sample sizes
        "n_total":  len(df),
        "n_AA":     int(len(minority)),
        "n_other":  int(len(majority)),

        # Descriptive group means
        "mean_score_AA":      round(score_AA,  3),
        "mean_score_other":   round(score_oth, 3),

        # Recidivism rates (rate ratio is a legitimate DIR)
        "recid_rate_AA":      round(rec_AA,  4),
        "recid_rate_other":   round(rec_oth, 4),
        "DIR_recidivism":     round(_safe_ratio(rec_AA, rec_oth), 3),

        # Mean-ratio "DIR" -- DESCRIPTIVE ONLY, not 4/5-rule comparable.
        "mean_score_ratio":   round(_safe_ratio(score_AA, score_oth), 3),
        "stat_parity_score":  round(score_AA - score_oth, 3),

        # Selection-rate DIR -- this IS what 4/5 rule is defined for.
        "high_risk_rate_AA":     round(high_AA,  4),
        "high_risk_rate_other":  round(high_oth, 4),
        "DIR_score_selection_rate":
            round(_safe_ratio(high_AA, high_oth), 3),
    }


def per_race_breakdown(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the paper's Table III layout: n + recidivism rate per race.

    Operates on the *raw* CSV after applying ProPublica's standard filters
    (so each race row uses the same denominator definition as the headline
    n=6,172 sample). Useful as a sanity check that you're computing rates
    on the right subset.
    """
    df = raw.copy()
    if "days_b_screening_arrest" in df.columns:
        df = df[df["days_b_screening_arrest"].abs() <= 30]
    if "is_recid" in df.columns:
        df = df[df["is_recid"] != -1]
    if "c_charge_degree" in df.columns:
        df = df[df["c_charge_degree"].isin(["F", "M"])]
    if "score_text" in df.columns:
        df = df[df["score_text"].notna() & (df["score_text"] != "N/A")]

    rows = []
    for race, sub in df.groupby("race"):
        n = len(sub)
        recid = int(sub["two_year_recid"].sum())
        rate  = recid / n if n else float("nan")
        rows.append({
            "race":        race,
            "n":           n,
            "recidivated": recid,
            "rate_pct":    round(rate * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def propublica_contingency(raw: pd.DataFrame, race: str) -> dict:
    """Reproduce ProPublica's contingency-table FP/FN rates for one race.

    ProPublica computes these on the n=7,214 sample (the full two-years CSV
    with only the score_text != "N/A" filter), not the n=6,172 logistic-
    regression sample. We mirror that here so the numbers come out exactly
    matching the published Black=44.85%/27.99% and White=23.45%/47.72%.

    Returns a dict with keys: FP_rate, FN_rate, PPV, NPV, n.
    """
    df = raw.copy()
    df = df[df["score_text"].notna() & (df["score_text"] != "N/A")]
    sub = df[df["race"] == race].copy()
    if len(sub) == 0:
        return {"n": 0, "FP_rate": float("nan"), "FN_rate": float("nan"),
                "PPV": float("nan"), "NPV": float("nan")}

    high = sub["score_text"] != "Low"           # Medium or High = "high risk"
    recid = sub["two_year_recid"] == 1

    # Confusion matrix with high-risk classification as the predictor
    tp = int(( high & recid).sum())
    fp = int(( high & ~recid).sum())
    fn = int((~high & recid).sum())
    tn = int((~high & ~recid).sum())

    fp_rate = fp / (fp + tn) if (fp + tn) else float("nan")
    fn_rate = fn / (fn + tp) if (fn + tp) else float("nan")
    ppv     = tp / (tp + fp) if (tp + fp) else float("nan")
    npv     = tn / (tn + fn) if (tn + fn) else float("nan")

    return {
        "n":       len(sub),
        "FP_rate": round(fp_rate * 100, 2),
        "FN_rate": round(fn_rate * 100, 2),
        "PPV":     round(ppv,     2),
        "NPV":     round(npv,     2),
    }


# --------------------------------------------------------------------------- #
# COMPAS Hypothesized Ground-Truth DAG (unchanged from previous version)
# --------------------------------------------------------------------------- #
# The TRUE causal graph for COMPAS is unknown. We write a "hypothesized" DAG
# encoding minimal uncontroversial domain assumptions: temporal/biological
# order in which variables come into existence and which edges are forbidden.
# Reviewers accept SHD against this hypothesized DAG if it's clearly labeled
# "hypothesized" and the assumptions are stated.

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

    # History -> recidivism
    ("Priors",       "Recidivism"),
    ("ChargeDegree", "Recidivism"),
    ("Age",          "Recidivism"),
    ("Sex",          "Recidivism"),
]

COMPAS_GROUND_TRUTH_EDGES_FAIR = [
    e for e in COMPAS_GROUND_TRUTH_EDGES
    if not (e[0] == "Race" and e[1] == "Score")
]


def get_compas_ground_truth(reference: str = "biased") -> list:
    """Return the hypothesized COMPAS ground-truth edge list.

    Parameters
    ----------
    reference : 'biased' or 'fair'
        - 'biased': includes the auditable Race -> Score edge.
        - 'fair' : excludes Race -> Score.
    """
    if reference == "biased":
        return list(COMPAS_GROUND_TRUTH_EDGES)
    if reference == "fair":
        return list(COMPAS_GROUND_TRUTH_EDGES_FAIR)
    raise ValueError(f"reference must be 'biased' or 'fair', got {reference!r}")


# --------------------------------------------------------------------------- #
# Correlation vs causal comparison figure (unchanged plotting; takes new
# values from the updated baseline_disparities dict)
# --------------------------------------------------------------------------- #

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
    """Two-panel comparison: correlation-based metrics (left) vs causal
    estimates (right). Every value is supplied by the caller so the figure
    always reflects the live run."""
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
        axL.text(i, v + 0.04, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=bar_color_corr)

    axL.axhline(0.8, color="black", linestyle="--", linewidth=1.5,
                label="4/5 rule threshold (0.8)")
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

    n_causal = len(causal_values)
    if n_causal == 0:
        greys = []
    elif n_causal == 1:
        greys = ["#5A5A5A"]
    else:
        greys = []
        for i in range(n_causal):
            level = int(200 - 140 * i / max(n_causal - 1, 1))
            greys.append(f"#{level:02x}{level:02x}{level:02x}")

    axR.bar(range(n_causal), causal_values, color=greys,
            edgecolor="black", linewidth=0.5)
    for i, v in enumerate(causal_values):
        axR.text(i, v + 0.008, f"{v:+.4f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    if n_causal >= 3:
        best_pair = None
        best_delta = float("inf")
        for i in range(1, n_causal - 1):
            d = abs(causal_values[i] - causal_values[i + 1])
            if d < best_delta:
                best_delta = d
                best_pair = (i, i + 1)
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

    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.97,
                        wspace=0.25)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)
    fig.text(0.5, 0.88, subtitle, ha="center", va="center", fontsize=10,
             style="italic", color="#444")
    save_figure_dual_format(fig, save_path)
    plt.close(fig)
    return fig


if __name__ == "__main__":
    # Sanity check when run directly
    raw = load_compas()
    df  = preprocess_compas(raw)
    print(f"\nPreprocessed COMPAS: n={len(df)} rows, {df.shape[1]} columns")
    print(df.head())
    print()
    print("=== Per-race breakdown (Table III style) ===")
    print(per_race_breakdown(raw).to_string(index=False))
    print()
    print("=== Baseline disparities ===")
    for k, v in baseline_disparities(df).items():
        print(f"  {k:30s} {v}")
    print()
    print("=== ProPublica-style contingency (n=7,214 sample) ===")
    for race in ["African-American", "Caucasian"]:
        c = propublica_contingency(raw, race)
        print(f"  {race:18s} n={c['n']:5d}  "
              f"FP={c['FP_rate']:5.2f}%  FN={c['FN_rate']:5.2f}%  "
              f"PPV={c['PPV']:.2f}  NPV={c['NPV']:.2f}")
