##!/usr/bin/env python3
"""
make_sensitivity_grid_SHDline.py -- regenerate the sensitivity grid with the
SHD line for the ICTAI 2026 research paper.

Grew out of the original Fig. 2 script. It builds the sensitivity grid as
PDF + PNG.

===========================================================================
NO FIGURE DATA IS HARDCODED IN THIS FILE
===========================================================================

Inputs (either one is sufficient; see --source)

  results/sensitivity_summary.csv   one row per (algorithm, n, beta)
      algorithm,n,beta,detection_rate,mean_shd,std_shd

  results/sensitivity_results.csv   one row per replicate
      algorithm,n,beta,<detected>,<shd>
      <detected> is any of DETECT_COLS: a 0/1, True/False, or yes/no flag
      <shd>      is any of SHD_COLS
      A 'rep'/'seed'/'run' column is ignored if present.

  With --source auto (the default) the summary is used when it exists and
  the replicate file is aggregated otherwise. --source results always
  re-aggregates; --validate compares the two and reports any disagreement.

Usage
  python make_sensitivity_grid_SHDline.py                  # build into figures/
  python make_sensitivity_grid_SHDline.py --check          # inputs only
  python make_sensitivity_grid_SHDline.py --source results --validate
  python make_sensitivity_grid_SHDline.py --annotate-shd   # SHD text in-cell
"""

# --------------------------------------------------------------------------
# Font configuration MUST happen before pyplot is imported.
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42      # 42 = TrueType. 3 = Type 3 = rejected.
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["text.usetex"] = False    # keep the script self-contained
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================================
# CONFIGURATION
# ==========================================================================

PATHS = {
    "sensitivity_summary": Path("results/sensitivity_summary.csv"),
    "sensitivity_results": Path("results/sensitivity_results.csv"),
}

OUTDIR = Path("figures")

# ---- page geometry -------------------------------------------------------
# IEEE two-column: 7.16 in is the full text width, 3.5 in a single column.
PAGE_W = 7.16
PAGE_H = 4.6
DPI_PNG = 400

# ---- palette -------------------------------------------------------------
BAR_C     = "#2E7D4F"   # detection-rate bars (green)
SHD_C     = "#B03A2E"   # mean-SHD line and its right-hand axis (brick red)
OUTLINE_C = "#B00020"   # outline on the main-setting bar (crimson)
EDGE_C    = "#1B1B1B"   # thin bar edge
GRID_C    = "#D7D7D7"
SUB_C     = "#444444"   # sub-title grey
ANN_C     = "#222222"   # in-cell SHD annotation

# ---- type scale (points) -------------------------------------------------
FS_TITLE = 8.0   # suptitle
FS_EDGE  = 5.8   # the legend line under the suptitle
FS_SUB   = 6.6   # per-column algorithm titles are FS_SUB + 0.9, bold
FS_AXIS  = 7.0   # axis labels
FS_TICK  = 6.3   # tick labels
FS_ANN   = 5.6   # in-cell SHD annotation

# ---- figure semantics ----------------------------------------------------
MAIN_BETA = 0.15        # the setting reported in the body of the paper
ANNOTATE_SHD = False    # in-cell "mean±sd" text; --annotate-shd turns it on
SHD_FLOOR = 4.0         # right axis never shrinks below this, so that a panel
                        # with near-zero SHD does not magnify its own noise

ALG_ORDER = ["PC", "FCI", "GES", "GRaSP", "ICA-LiNGAM", "DirectLiNGAM"]

# Replicate-file column names we are willing to recognise, in priority order.
DETECT_COLS = ["detected", "detection", "detect", "found", "recovered",
               "race_loan_detected", "race_to_loan", "edge_detected",
               "true_edge_detected", "hit"]
SHD_COLS    = ["shd", "structural_hamming_distance", "shd_value"]

SUMMARY_COLS = ["algorithm", "n", "beta", "detection_rate",
                "mean_shd", "std_shd"]
SUMMARY_SCHEMA = ",".join(SUMMARY_COLS)
RESULTS_SCHEMA = "algorithm,n,beta,<detected>,<shd>  (one row per replicate)"


# ==========================================================================
# input helpers -- every one of these fails loudly rather than inventing data
# ==========================================================================
class MissingInput(Exception):
    def __init__(self, path, schema):
        super().__init__(f"missing input: {path}\n    expected: {schema}")
        self.path, self.schema = path, schema


def need_csv(path, schema, **kw):
    if not Path(path).exists():
        raise MissingInput(path, schema)
    return pd.read_csv(path, **kw)


def _pick_column(df, candidates, role, path):
    """First candidate present in df, matched case-insensitively."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise SystemExit(
        f"{path}: no {role} column found.\n"
        f"    looked for: {', '.join(candidates)}\n"
        f"    present:    {', '.join(df.columns)}")


def _as_indicator(s, path, col):
    """Coerce a detection flag to 0/1 without silently swallowing junk."""
    if s.dtype == bool:
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        bad = set(pd.unique(s.dropna())) - {0, 1, 0.0, 1.0}
        if bad:
            raise SystemExit(f"{path}: column '{col}' must be 0/1, "
                             f"saw {sorted(bad)[:5]}")
        return s.astype(float)
    m = {"true": 1.0, "t": 1.0, "yes": 1.0, "y": 1.0, "1": 1.0,
         "false": 0.0, "f": 0.0, "no": 0.0, "n": 0.0, "0": 0.0}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        bad = sorted(set(s[out.isna()].astype(str)))[:5]
        raise SystemExit(f"{path}: column '{col}' has unparseable "
                         f"values {bad}")
    return out


def aggregate_results(path=None):
    """Collapse the replicate table to the per-cell summary schema.

    std is the sample standard deviation (ddof=1), which is what pandas'
    .std() gives and what the published table was built with.
    """
    path = Path(path or PATHS["sensitivity_results"])
    df = need_csv(path, RESULTS_SCHEMA)
    for key in ("algorithm", "n", "beta"):
        if key not in {c.lower() for c in df.columns}:
            raise SystemExit(f"{path}: missing key column '{key}'")
    df = df.rename(columns={c: c.lower() for c in df.columns})

    det = _pick_column(df, DETECT_COLS, "detection-flag", path)
    shd = _pick_column(df, SHD_COLS, "SHD", path)
    df["_det"] = _as_indicator(df[det], path, det)

    g = df.groupby(["algorithm", "n", "beta"], as_index=False)
    out = g.agg(detection_rate=("_det", "mean"),
                mean_shd=(shd, "mean"),
                std_shd=(shd, "std"),
                n_reps=("_det", "size"))
    # A single replicate makes std NaN; that is a data problem, not a plot
    # problem, so say so rather than drawing a zero.
    if out.std_shd.isna().any():
        thin = out[out.std_shd.isna()][["algorithm", "n", "beta", "n_reps"]]
        raise SystemExit(f"{path}: cells with fewer than 2 replicates:\n"
                         f"{thin.to_string(index=False)}")
    reps = sorted(out.n_reps.unique())
    print(f"  aggregated {len(df):,} replicate rows -> {len(out)} cells "
          f"({'/'.join(str(r) for r in reps)} reps per cell)")
    return out[SUMMARY_COLS]


def load_sensitivity(source="auto", validate=False):
    """Return the per-cell summary table, from whichever input is available."""
    summ_p, res_p = PATHS["sensitivity_summary"], PATHS["sensitivity_results"]

    if source == "summary":
        df = need_csv(summ_p, SUMMARY_SCHEMA)
        src = summ_p
    elif source == "results":
        df, src = aggregate_results(res_p), res_p
    else:                                     # auto
        if summ_p.exists():
            df, src = need_csv(summ_p, SUMMARY_SCHEMA), summ_p
        elif res_p.exists():
            df, src = aggregate_results(res_p), res_p
        else:
            raise MissingInput(f"{summ_p} (or {res_p})",
                               f"{SUMMARY_SCHEMA}  /  {RESULTS_SCHEMA}")

    missing = [c for c in SUMMARY_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{src}: missing {missing}")
    print(f"  source: {src}")

    if validate and summ_p.exists() and res_p.exists():
        _validate(need_csv(summ_p, SUMMARY_SCHEMA), aggregate_results(res_p))
    return df


def _validate(summary, derived, tol=5e-3):
    """Cross-check the shipped summary against the replicate file."""
    keys = ["algorithm", "n", "beta"]
    m = summary.merge(derived, on=keys, how="outer",
                      suffixes=("_summary", "_derived"), indicator=True)
    only = m[m._merge != "both"]
    if len(only):
        print(f"  VALIDATE: {len(only)} cell(s) in only one file:")
        print(only[keys + ["_merge"]].to_string(index=False))
    both, bad = m[m._merge == "both"], 0
    for col in ("detection_rate", "mean_shd", "std_shd"):
        d = (both[f"{col}_summary"] - both[f"{col}_derived"]).abs()
        off = both[d > tol]
        if len(off):
            bad += len(off)
            print(f"  VALIDATE: {len(off)} cell(s) disagree on {col} "
                  f"(max {d.max():.4f})")
            print(off[keys + [f"{col}_summary",
                              f"{col}_derived"]].head(10).to_string(index=False))
    if not bad and not len(only):
        print(f"  VALIDATE: summary and replicates agree "
              f"({len(both)} cells, tol={tol})")


def save(fig, stem, note=""):
    """PDF for the paper, PNG for looking at it. Both, always."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUTDIR / f"{stem}.pdf", OUTDIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI_PNG)
    plt.close(fig)
    print(f"  wrote {pdf}{note}")
    print(f"  wrote {png}")


# ==========================================================================
# Fig. 2 -- sensitivity grid
# ==========================================================================
def fmt_shd(mean, std):
    """One decimal when either component is sub-integer, else integer.

    '.0f' on (0.35, 0.49) yields '0+/-0', which reads as perfect structural
    recovery with zero variance. Neither is true.
    """
    if mean < 1 or std < 1:
        return f"{mean:.1f}±{std:.1f}"
    return f"{mean:.0f}±{std:.0f}"


def fig2(args):
    df = load_sensitivity(args.source, args.validate)
    need = set(SUMMARY_COLS)
    if need - set(df.columns):
        raise SystemExit(f"{PATHS['sensitivity_summary']}: missing "
                         f"{sorted(need - set(df.columns))}")

    algorithms = [a for a in ALG_ORDER if a in set(df.algorithm)]
    extra = sorted(set(df.algorithm) - set(ALG_ORDER))
    if extra:
        print(f"  note: not in ALG_ORDER, so not drawn: {extra}")
    if not algorithms:
        raise SystemExit("no recognised algorithms; check the algorithm "
                         f"column against ALG_ORDER={ALG_ORDER}")
    sizes = sorted(df.n.unique())
    betas = sorted(df.beta.unique())

    fig, axes = plt.subplots(len(sizes), len(algorithms),
                             figsize=(PAGE_W, PAGE_H), sharex=True,
                             sharey=True, squeeze=False)
    shd_max = max(SHD_FLOOR, float(df.mean_shd.max()) * 1.05)
    rows_out = []

    for i, n in enumerate(sizes):
        for j, alg in enumerate(algorithms):
            ax = axes[i][j]
            cell = df[(df.algorithm == alg) & (df.n == n)].set_index("beta")
            if cell.index.duplicated().any():
                dup = sorted(cell.index[cell.index.duplicated()].unique())
                raise SystemExit(f"duplicate rows for algorithm={alg}, "
                                 f"n={n}, beta={dup}")
            rates, means, stds = [], [], []
            for b in betas:
                if b not in cell.index:
                    raise SystemExit(f"{PATHS['sensitivity_summary']}: no row "
                                     f"for algorithm={alg}, n={n}, beta={b}")
                r = cell.loc[b]
                rates.append(float(r.detection_rate))
                means.append(float(r.mean_shd))
                stds.append(float(r.std_shd))
                rows_out.append({"algorithm": alg, "n": n, "beta": b,
                                 "detection_rate": float(r.detection_rate),
                                 "mean_shd": float(r.mean_shd),
                                 "std_shd": float(r.std_shd),
                                 "shd_label": fmt_shd(float(r.mean_shd),
                                                      float(r.std_shd))})

            x = list(range(len(betas)))
            bars = ax.bar(x, rates, width=0.74, color=BAR_C,
                          edgecolor=EDGE_C, linewidth=0.3, zorder=3)
            for bar, r, b in zip(bars, rates, betas):
                bar.set_alpha(0.25 + 0.75 * r)
                if abs(b - MAIN_BETA) < 1e-9:
                    bar.set_edgecolor(OUTLINE_C)
                    bar.set_linewidth(1.0)

            ax2 = ax.twinx()
            ax2.plot(x, means, color=SHD_C, linewidth=0.8, marker="o",
                     markersize=1.8, zorder=4)
            ax2.set_ylim(0, shd_max)
            if j == len(algorithms) - 1:
                ax2.tick_params(axis="y", labelsize=FS_TICK, length=1.5, pad=1,
                                colors=SHD_C)
                ax2.set_ylabel("mean SHD", fontsize=FS_AXIS, color=SHD_C,
                               labelpad=2)
            else:
                ax2.tick_params(axis="y", length=0, labelright=False)
                ax2.set_yticklabels([])

            if args.annotate_shd:
                for xi, (r, m, sd) in enumerate(zip(rates, means, stds)):
                    ax.text(xi, min(r + 0.04, 0.72), fmt_shd(m, sd),
                            ha="center", va="bottom", rotation=90,
                            fontsize=FS_ANN, color=ANN_C, zorder=5)

            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks(x)
            ax.set_xticklabels([f"{b:.2f}" for b in betas], fontsize=FS_TICK,
                               rotation=90)
            ax.tick_params(axis="y", labelsize=FS_TICK, length=1.5, pad=1)
            ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5,
                    zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
                ax2.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_linewidth(0.4)
            if i == 0:
                ax.set_title(alg, fontsize=FS_SUB + 0.9, fontweight="bold",
                             pad=4)
            if j == 0:
                ax.set_ylabel(f"$n$={n:,}\ndetection rate", fontsize=FS_AXIS,
                              labelpad=2)
            if i == len(sizes) - 1:
                ax.set_xlabel(r"$\beta$", fontsize=FS_AXIS, labelpad=1)

    fig.suptitle(r"Sensitivity analysis: Race$\rightarrow$Loan detection rate "
                 r"and SHD", fontsize=FS_TITLE, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "bar height and fill = detection rate; red line = mean SHD "
             "(right axis); red outline = main setting "
             rf"$\beta$ = {MAIN_BETA:g}",
             ha="center", fontsize=FS_EDGE, color=SUB_C)
    fig.subplots_adjust(left=0.075, right=0.945, top=0.895, bottom=0.105,
                        wspace=0.16, hspace=0.22)
    save(fig, "fig2_sensitivity_grid", f"  ({len(rows_out)} cells)")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(OUTDIR / "fig2_shd_table.csv", index=False)
    print(f"  wrote {OUTDIR/'fig2_shd_table.csv'}")


# ==========================================================================
# driver
# ==========================================================================
BUILDERS = {"fig2": fig2}

CHECKLIST = [
    ("fig2", "sensitivity_summary", SUMMARY_SCHEMA),
    ("fig2", "sensitivity_results", RESULTS_SCHEMA),
]


def do_check():
    print("input check\n" + "-" * 66)
    present = 0
    for fig, key, schema in CHECKLIST:
        p = PATHS[key]
        ok = Path(p).exists()
        present += ok
        print(f"  [{'ok' if ok else '--'}] {fig}  {p}")
        if not ok:
            print(f"          expected: {schema}")
    if present:
        print(f"\n  {present} of {len(CHECKLIST)} input(s) present; "
              f"either one is enough to build fig2.")
        return 0
    print("\n  no sensitivity input found.")
    return 1


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate the sensitivity grid figure for the ICTAI "
                    "paper.")
    ap.add_argument("--only", default="",
                    help="comma-separated subset, e.g. fig2")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--source", choices=["auto", "summary", "results"],
                    default="auto",
                    help="auto (default) prefers sensitivity_summary.csv and "
                         "falls back to aggregating sensitivity_results.csv")
    ap.add_argument("--validate", action="store_true",
                    help="when both inputs exist, report any cell where the "
                         "summary disagrees with the replicates")
    ap.add_argument("--annotate-shd", action="store_true",
                    help="print mean±sd SHD inside each cell")
    ap.add_argument("--main-beta", type=float, default=MAIN_BETA,
                    help="beta given the red outline (default %(default)s)")
    ap.add_argument("--dpi", type=int, default=DPI_PNG,
                    help="PNG resolution (the PDF stays vector)")
    ap.add_argument("--strict", action="store_true",
                    help="a missing input is fatal instead of a skip")
    ap.add_argument("--check", action="store_true",
                    help="report which inputs are present, build nothing")
    args = ap.parse_args()

    globals()["OUTDIR"] = Path(args.outdir)
    globals()["MAIN_BETA"] = args.main_beta
    globals()["DPI_PNG"] = args.dpi

    if args.check:
        sys.exit(do_check())

    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or \
             list(BUILDERS)
    unknown = [w for w in wanted if w not in BUILDERS]
    if unknown:
        raise SystemExit(f"unknown figure(s): {unknown}. "
                         f"Choose from {list(BUILDERS)}")

    built, skipped = 0, []
    for name in wanted:
        print(f"{name}:")
        try:
            BUILDERS[name](args)
            built += 1
        except MissingInput as exc:
            if args.strict:
                raise SystemExit(f"  {exc}")
            print(f"  skipped -- {exc}")
            skipped.append(name)

    print(f"\nbuilt {built} figure(s)"
          + (f", skipped {', '.join(skipped)}" if skipped else ""))
    print("verify fonts:  pdffonts figures/*.pdf   "
          "-> no row should say 'Type 3'")


if __name__ == "__main__":
    main()