#!/usr/bin/env python3
"""
make_fig2.py -- regenerate EVERY figure for ICTAI 2026 paper 582.

Grew out of the original Fig. 2 script. It now builds the sensitivity grid AND
the causal-graph figures, all in matplotlib, all as PDF + PNG.

===========================================================================
NO FIGURE DATA IS HARDCODED IN THIS FILE
===========================================================================
Every number and every edge drawn comes from a file on disk that your
pipeline produced: the sensitivity summary, the generated synthetic datasets,
the saved SCM specification, and the saved algorithm output matrices.

If an input is missing the script STOPS and tells you the exact path and
schema it wanted. It never falls back to a stored value, so a figure can
never silently disagree with the experiment behind it.

The only things declared here are semantic annotations that are not data --
which variables count as protected attributes, proxies, mediators, outcomes
(NODE_ROLES), and which edges the prose discusses and therefore get a printed
coefficient (LABEL_EDGES / FLAG_EDGES). Those are editorial choices about the
figure, not measurements. Every printed coefficient is still read from a file.

===========================================================================
INPUTS
===========================================================================
Fig. 1  ground-truth SCM
  PATHS["scm_spec"]        from,to,weight[,planted]     <- your generator
  PATHS["synth_biased"]    the generated Dataset A      <- used to verify
  PATHS["synth_unbiased"]  the generated Dataset B      <- only if SYNTH_PANELS==2

  The spec is the ground truth: a specification, not an estimate, so it
  legitimately comes from the data-generating code. Export it once, e.g.

      import csv
      with open("results/scm_spec.csv", "w", newline="") as fh:
          w = csv.writer(fh); w.writerow(["from","to","weight","planted"])
          for (src, dst), coef in SCM_EDGES.items():
              w.writerow([src, dst, coef, int((src,dst) == ("Race","Loan"))])

  On every run the script re-estimates those coefficients by OLS from the
  generated data and prints a spec-vs-data comparison. A mismatch above
  OLS_TOLERANCE is reported loudly. That is the check that keeps Fig. 1
  honest.

Fig. 2  sensitivity grid
  PATHS["sensitivity"]     algorithm,n,beta,detection_rate,mean_shd,std_shd

Fig. 3  DirectLiNGAM on the synthetic biased dataset
  PATHS["synth_dlingam_B"] square LiNGAM adjacency, B[i][j] = coef on j -> i
  PATHS["synth_varnames"]  one variable name per line, matrix column order

Fig. 4  algorithm comparison on COMPAS
  PATHS["compas_graphs"][alg]  square causal-learn GeneralGraph.graph matrix
                               or a LiNGAM B matrix (see COMPAS_GRAPH_KIND)
  PATHS["compas_varnames"]     one variable name per line

Fig. 5  DirectLiNGAM on COMPAS
  PATHS["compas_dlingam_B"]    square LiNGAM adjacency
  PATHS["compas_frame"]        the analysis frame, for the reported n

Save the matrices once, at the end of your existing runs:

    np.savetxt("results/compas_pc_graph.csv",  pc_G.graph,   delimiter=",")
    np.savetxt("results/compas_fci_graph.csv", fci_G.graph,  delimiter=",")
    np.savetxt("results/compas_dlingam_B.csv",
               dlingam_model.adjacency_matrix_, delimiter=",")
    with open("results/compas_varnames.txt", "w") as fh:
        fh.write("\\n".join(column_names))

===========================================================================
USAGE
===========================================================================
    python make_fig2.py --check              # what is present, what is missing
    python make_fig2.py                      # build every figure it can
    python make_fig2.py --only fig1,fig2     # build a subset
    python make_fig2.py --synth-panels 1     # single-panel Fig. 1 (see toggle)
    python make_fig2.py --strict             # any missing input is fatal

===========================================================================
WHAT CHANGED FROM THE SUBMITTED VERSION
===========================================================================
1. pdf.fonttype / ps.fonttype = 42, set BEFORE pyplot is imported (this is
   the part that matters -- setting them afterwards has no effect). The
   submitted PDF embedded DejaVu Sans as Type 3 subsets, which IEEE PDF
   eXpress rejects.
2. Figures are authored at their final printed size. The old Fig. 2 was drawn
   at 16.5 in and scaled into a 3.5 in column, so its 6.4 pt annotations
   printed near 1.4 pt. Reviewers 4 and 5 both objected to figure legibility.
3. bbox_inches="tight" is gone. It rescales the figure and every type size
   with it, which is why the sizes specified were not the sizes that printed.
4. The DAGs use a deterministic layered layout computed from the graph's own
   topology, not a spring layout, so there is no dead white space and the
   figure is identical on every run.
5. SYNTH_PANELS toggles Fig. 1 between the biased panel alone and the
   biased/unbiased pair.
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
from matplotlib.patches import Ellipse, FancyArrowPatch

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================================
# CONFIGURATION
# ==========================================================================

PATHS = {
    "sensitivity":      Path("results/sensitivity_summary.csv"),
    "scm_spec":         Path("results/scm_spec.csv"),
    "synth_biased":     Path("data/synthetic_biased.csv"),
    "synth_unbiased":   Path("data/synthetic_unbiased.csv"),
    "synth_dlingam_B":  Path("results/synth_dlingam_B.csv"),
    "synth_varnames":   Path("results/synth_varnames.txt"),
    "compas_dlingam_B": Path("results/compas_dlingam_B.csv"),
    "compas_varnames":  Path("results/compas_varnames.txt"),
    "compas_frame":     Path("results/compas_analysis_frame.csv"),
}

# Fig. 4 panels: label -> (path, kind). kind is "causal-learn" or "lingam".
COMPAS_GRAPHS = {
    "PC":            (Path("results/compas_pc_graph.csv"),      "causal-learn"),
    "FCI":           (Path("results/compas_fci_graph.csv"),     "causal-learn"),
    "GES":           (Path("results/compas_ges_graph.csv"),     "causal-learn"),
    "GRaSP":         (Path("results/compas_grasp_graph.csv"),   "causal-learn"),
    "ICA-LiNGAM":    (Path("results/compas_icalingam_B.csv"),   "lingam"),
    "DirectLiNGAM":  (Path("results/compas_dlingam_B.csv"),     "lingam"),
}
# Which of those to draw in Fig. 4. Six panels in one float is what made the
# submitted figure illegible; three is readable at the same page cost.
COMPAS_PANELS = ["FCI", "GRaSP", "DirectLiNGAM"]

OUTDIR = Path("figures")

# ---- Fig. 1 panel toggle -------------------------------------------------
# 1 = biased SCM only (Dataset A). 2 = biased and unbiased side by side.
# Overridable with --synth-panels.
SYNTH_PANELS = 1

# ---- semantic annotation (not data) --------------------------------------
# Which role each variable plays in the fairness argument. This is domain
# knowledge from the paper, not a measurement; it only sets the fill colour.
NODE_ROLES = {
    # synthetic loan-approval SCM
    "Race": "protected", "Gender": "protected",
    "ZIP": "proxy", "ZIP Code": "proxy",
    "Income": "mediator", "CreditSc": "mediator", "Credit Score": "mediator",
    "Education": "covariate", "SES": "latent",
    "Loan": "outcome", "Loan approval": "outcome",
    # COMPAS
    "Sex": "protected", "Age": "covariate",
    "JuvFel": "mediator", "JuvFelony": "mediator", "Juvenile Felony": "mediator",
    "JuvMisd": "mediator", "Juvenile Misdemeanor": "mediator",
    "Charge": "covariate", "ChargeDegree": "covariate", "Charge Degree": "covariate",
    "Priors": "proxy", "Prior Convictions": "proxy",
    "Score": "outcome", "COMPAS Score": "outcome", "DecileScore": "outcome",
    "Recid": "outcome", "Recidivism": "outcome",
}
DEFAULT_ROLE = "covariate"

# Shortened display labels. Purely cosmetic.
DISPLAY = {
    "CreditSc": "Credit\nScore", "Credit Score": "Credit\nScore",
    "JuvFel": "Juv\nFel", "JuvFelony": "Juv\nFel",
    "JuvMisd": "Juv\nMisd", "ChargeDegree": "Charge\nDegree",
    "Charge Degree": "Charge\nDegree", "Prior Convictions": "Priors",
    "COMPAS Score": "COMPAS\nScore", "DecileScore": "COMPAS\nScore",
    "Recidivism": "Recid.", "Education": "Educ.",
}

# Edges whose coefficient is printed. The VALUES come from the data files;
# this only says which ones the prose discusses. Everything else is drawn
# unlabelled -- eighteen 5 pt labels is what made the submitted Fig. 5
# unreadable.
LABEL_EDGES = {
    "fig3": [("Race", "Loan")],
    "fig5": [("Race", "Score"), ("Race", "Priors"), ("Priors", "Score")],
}
# Edges drawn red/dashed as the flagged discrimination paths.
FLAG_EDGES = {
    "fig1": [("Race", "Loan")],
    "fig3": [("Race", "Loan")],
    "fig4": [("Race", "Score"), ("Score", "Race"), ("Score", "Sex")],
    "fig5": [("Race", "Score"), ("Race", "Priors")],
}

LINGAM_THRESHOLD = 0.01     # |coefficient| below this is treated as no edge
OLS_TOLERANCE = 0.05        # spec-vs-data gap worth flagging. At n=5,000 the
                            # OLS sampling error on a single coefficient is
                            # already ~0.02, so a tighter bound just cries
                            # wolf. This catches sign flips and lost decimal
                            # points, which are the errors that matter.

# ---- appearance ----------------------------------------------------------
COL_W, PAGE_W = 3.5, 7.16   # IEEE single column / two-column float, inches

ROLE_FILL = {
    "protected": "#A9CCE3", "proxy": "#F5B041", "mediator": "#A9DFBF",
    "covariate": "#D0D3D4", "outcome": "#F5B7B1", "latent": "#FFFFFF",
}
ROLE_CAPTION = {
    "protected": "Protected attribute", "proxy": "Proxy discriminator",
    "mediator": "Mediator", "covariate": "Covariate",
    "outcome": "Outcome", "latent": "Latent (unobserved)",
}
EDGE_C, MUTED_C, FLAG_C = "#2B2B2B", "#95A0A8", "#C0392B"
BAR_C, SHD_C, OUTLINE_C = "#2E7D4F", "#B03A2E", "#B00020"

FS_TITLE, FS_SUB, FS_NODE, FS_EDGE, FS_AXIS, FS_TICK = 8.0, 6.6, 6.2, 5.8, 7.0, 6.2

MAIN_BETA = 0.15
ANNOTATE_SHD = False        # True restores the per-bar rotated SHD annotations


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


def need_matrix(path, schema):
    if not Path(path).exists():
        raise MissingInput(path, schema)
    m = np.loadtxt(path, delimiter=",", ndmin=2)
    if m.shape[0] != m.shape[1]:
        raise SystemExit(f"{path}: expected a square matrix, got {m.shape}")
    return m


def need_names(path, n_expected=None):
    if not Path(path).exists():
        raise MissingInput(path, "one variable name per line, matrix column order")
    names = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if n_expected is not None and len(names) != n_expected:
        raise SystemExit(f"{path}: {len(names)} names but the matrix is "
                         f"{n_expected}x{n_expected}")
    return names


def decode_causal_learn(m, names):
    """causal-learn GeneralGraph.graph -> [(src, dst, kind)].

    Encoding: g[j,i]=1 and g[i,j]=-1 is i->j; both -1 is undirected;
    both 1 is bidirected (latent common cause).
    """
    out, n = [], len(names)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = m[i, j], m[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                out.append((names[i], names[j], "directed"))
            elif b == -1 and a == 1:
                out.append((names[j], names[i], "directed"))
            elif a == 1 and b == 1:
                out.append((names[i], names[j], "bidirected"))
            else:
                out.append((names[i], names[j], "undirected"))
    return out


def decode_lingam(m, names, threshold=LINGAM_THRESHOLD):
    """LiNGAM adjacency -> [(src, dst, kind, weight)]; B[i,j] is coef on j->i."""
    out = []
    for i, dst in enumerate(names):
        for j, src in enumerate(names):
            if i != j and abs(m[i, j]) >= threshold:
                out.append((src, dst, "directed", float(m[i, j])))
    return out


# ==========================================================================
# deterministic layered DAG layout
# ==========================================================================
def layered_positions(nodes, edges, x_gap=1.0, y_gap=1.0, sweeps=8):
    """Longest-path layering, then barycentre ordering. Fully deterministic.

    No spring layout and no randomness: the same graph always produces the
    same picture, which is part of what makes the figure reproducible.
    """
    succ, pred = defaultdict(list), defaultdict(list)
    for e in edges:
        a, b = e[0], e[1]
        if a != b:
            succ[a].append(b)
            pred[b].append(a)

    layer = {v: 0 for v in nodes}
    for _ in range(len(nodes) + 1):
        changed = False
        for v in nodes:
            for w in succ[v]:
                if layer[w] < layer[v] + 1:
                    layer[w] = layer[v] + 1
                    changed = True
        if not changed:
            break

    seed = {"protected": 0, "proxy": 1, "mediator": 2,
            "covariate": 3, "outcome": 4, "latent": 5}
    buckets = defaultdict(list)
    for v in sorted(nodes, key=lambda v: (
            seed.get(NODE_ROLES.get(v, DEFAULT_ROLE), 9), v)):
        buckets[layer[v]].append(v)
    order = {L: list(vs) for L, vs in buckets.items()}

    def ys():
        out = {}
        for L, vs in order.items():
            top = (len(vs) - 1) * y_gap / 2.0
            for k, v in enumerate(vs):
                out[v] = top - k * y_gap
        return out

    layers = sorted(order)
    for sweep in range(sweeps):
        y = ys()
        forward = sweep % 2 == 0
        seq = layers[1:] if forward else list(reversed(layers[:-1]))
        ref = pred if forward else succ
        for L in seq:
            def bary(v):
                nb = [y[u] for u in ref[v] if u in y]
                return -(sum(nb) / len(nb)) if nb else -y[v]
            order[L] = sorted(order[L], key=lambda v: (bary(v), v))

    y = ys()
    return {v: (layer[v] * x_gap, y[v]) for v in nodes}, layer


def _bezier_point(p0, p2, rad, t):
    """Point at parameter t on the quadratic Bezier that arc3 draws."""
    x0, y0 = p0
    x2, y2 = p2
    mx, my = (x0 + x2) / 2.0, (y0 + y2) / 2.0
    dx, dy = x2 - x0, y2 - y0
    cx, cy = mx + rad * dy, my - rad * dx
    u = 1.0 - t
    return (u * u * x0 + 2 * u * t * cx + t * t * x2,
            u * u * y0 + 2 * u * t * cy + t * t * y2)


def _outside(q, pos, rx, ry, margin=1.0):
    """Smallest normalised distance from q to any node ellipse (>1 is clear)."""
    best = 1e9
    for cx, cy in pos.values():
        d = np.hypot((q[0] - cx) / (rx * margin), (q[1] - cy) / (ry * margin))
        best = min(best, d)
    return best


def _label_point(p0, p2, rad, pos, rx, ry, crowded=False, fan=0, lift=0.0):
    """Midpoint of the drawn curve, nudged along it if it lands on a node.

    When several edges converge on one node their midpoints pile up, so for a
    crowded target the label is biased back toward the source.
    """
    if abs(rad) < 1e-9 and lift:
        # straight edge between neighbouring layers: there is no room on the
        # line, so lift the label clear of it
        q = _bezier_point(p0, p2, rad, 0.5)
        dx, dy = p2[0] - p0[0], p2[1] - p0[1]
        L = max(np.hypot(dx, dy), 1e-9)
        return (q[0] - dy / L * lift, q[1] + dx / L * lift)

    shift = 0.09 * fan
    order = ((0.36, 0.30, 0.44, 0.24, 0.52, 0.60) if crowded
             else (0.50, 0.42, 0.58, 0.34, 0.66, 0.28, 0.72))
    order = [min(max(t + shift, 0.15), 0.85) for t in order]
    best, best_clear = None, -1.0
    for t in order:
        q = _bezier_point(p0, p2, rad, t)
        clear = _outside(q, pos, rx, ry, margin=1.35)
        if clear > 1.0:
            return q
        if clear > best_clear:
            best, best_clear = q, clear
    return best


def _curvature(src, dst, pos, layer, seen, y_center):
    """Arc enough to clear the nodes an edge flies over; fan out parallels.

    arc3 with rad > 0 bulges to the left of the travel direction, so for a
    left-to-right edge a positive rad bulges upward. A long edge is therefore
    routed AWAY from the vertical centre of the drawing, which keeps it from
    ploughing through the nodes it spans.
    """
    span = abs(layer[dst] - layer[src])
    key = tuple(sorted((src, dst)))
    k = seen[key]
    seen[key] += 1
    y0, y1 = pos[src][1], pos[dst][1]

    if span <= 1:
        if k == 0:
            return 0.0
        base = 0.15 * ((k + 1) // 2)
        return base if k % 2 == 1 else -base

    base = 0.16 + 0.06 * (span - 1) + 0.09 * k
    mid_y = (y0 + y1) / 2.0
    sign = -1.0 if mid_y > y_center else 1.0      # bulge outward, not inward
    if abs(mid_y - y_center) < 1e-9:
        sign = 1.0 if k % 2 == 0 else -1.0
    return sign * base


def draw_dag(ax, nodes, edges, *, labels=None, flags=(), title=None,
             node_w=0.86, node_h=0.46, fs_node=FS_NODE, fs_edge=FS_EDGE,
             x_gap=1.0, y_gap=1.0, aspect=None):
    """edges: [(src, dst, kind)]. kind in {directed, bidirected, undirected}."""
    labels = labels or {}
    flagset = {tuple(e) for e in flags}

    # First pass fixes the layering; then choose the vertical gap so the
    # drawing's own aspect ratio matches the axes it has to live in. Without
    # this the DAG is letterboxed and the nodes shrink -- which is exactly
    # what the reviewers were reading as "too small".
    pos, layer = layered_positions(nodes, edges, x_gap=x_gap, y_gap=1.0)
    n_layers = max(layer.values()) + 1
    per_layer = defaultdict(int)
    for v in nodes:
        per_layer[layer[v]] += 1
    tallest = max(per_layer.values())
    if aspect and n_layers > 1 and tallest > 1:
        width_u = (n_layers - 1) * x_gap
        y_gap = max(node_h * 2.05, (width_u / aspect) / (tallest - 1))
    pos, layer = layered_positions(nodes, edges, x_gap=x_gap, y_gap=y_gap)

    indeg = defaultdict(int)
    for e in edges:
        indeg[e[1]] += 1
    y_center = (max(p[1] for p in pos.values())
                + min(p[1] for p in pos.values())) / 2.0
    seen = defaultdict(int)
    out_seen = defaultdict(int)
    # draw unlabelled structure first so labelled/flagged edges sit on top
    ranked = sorted(edges, key=lambda e: ((e[0], e[1]) in flagset,
                                          (e[0], e[1]) in labels))
    for e in ranked:
        src, dst, kind = e[0], e[1], e[2]
        is_flag = (src, dst) in flagset
        lab = labels.get((src, dst))
        fan = out_seen[src]
        out_seen[src] += 1
        rad = _curvature(src, dst, pos, layer, seen, y_center)

        style = {"bidirected": "<|-|>", "undirected": "-"}.get(kind, "-|>")
        if is_flag:
            color, lw, ls, z = FLAG_C, 1.25, (0, (3.0, 1.9)), 4
        elif lab is not None:
            color, lw, ls, z = EDGE_C, 0.9, "-", 3
        else:
            color, lw, ls, z = MUTED_C, 0.6, "-", 2

        # trim the chord to the node boundaries, then keep the same curvature
        rx, ry = node_w / 2.0, node_h / 2.0
        a_pt = _shrink_end(pos[src], pos[dst], rad, rx, ry, at_start=True)
        b_pt = _shrink_end(pos[src], pos[dst], rad, rx, ry, at_start=False)
        ax.add_patch(FancyArrowPatch(
            a_pt, b_pt, arrowstyle=style,
            connectionstyle=f"arc3,rad={rad:.4f}",
            mutation_scale=6.5, linewidth=lw, linestyle=ls, color=color,
            shrinkA=0, shrinkB=0,
            joinstyle="round", capstyle="round", zorder=z))

        if lab is not None:
            px, py = _label_point(pos[src], pos[dst], rad, pos, rx, ry,
                                  crowded=indeg[dst] >= 3, fan=fan,
                                  lift=node_h * 0.72)
            ax.text(px, py, lab, ha="center", va="center", fontsize=fs_edge,
                    color=FLAG_C if is_flag else EDGE_C,
                    fontweight="bold" if is_flag else "normal", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.13", fc="white",
                              ec="none", alpha=0.9))

    for v in nodes:
        role = NODE_ROLES.get(v, DEFAULT_ROLE)
        x, y = pos[v]
        ax.add_patch(Ellipse((x, y), node_w, node_h,
                             facecolor=ROLE_FILL[role],
                             edgecolor=MUTED_C if role == "latent" else "#1B1B1B",
                             linewidth=0.7, zorder=5,
                             linestyle="--" if role == "latent" else "-"))
        ax.text(x, y, DISPLAY.get(v, v), ha="center", va="center",
                fontsize=fs_node, fontweight="bold", zorder=6,
                color=MUTED_C if role == "latent" else "#111111",
                linespacing=0.95)

    xs = [p[0] for p in pos.values()]
    ys_ = [p[1] for p in pos.values()]
    pad = max(node_w * 0.72, node_h * 1.5)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys_) - pad, max(ys_) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=FS_SUB, fontweight="bold", pad=3)
    return pos


def _shrink_end(p0, p2, rad, rx, ry, at_start):
    """Walk along the curve until it leaves the node ellipse at that end."""
    ax_, ay = p0 if at_start else p2
    ts = np.linspace(0.0, 0.5, 60) if at_start else np.linspace(1.0, 0.5, 60)
    for t in ts:
        q = _bezier_point(p0, p2, rad, float(t))
        if np.hypot((q[0] - ax_) / rx, (q[1] - ay) / ry) >= 1.0:
            return q
    return _bezier_point(p0, p2, rad, 0.5)


ROLE_ORDER = ["protected", "proxy", "mediator", "covariate", "outcome",
              "latent"]


def role_legend(fig, roles_used, *, y=0.035, extra=()):
    roles_used = [r for r in ROLE_ORDER if r in set(roles_used)]
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ROLE_FILL[r], edgecolor="#1B1B1B",
                     linewidth=0.6, label=ROLE_CAPTION[r]) for r in roles_used]
    handles += list(extra)
    ncol = 2 if fig.get_figwidth() < 4.5 else 4
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), ncol),
               fontsize=FS_EDGE - 0.4, frameon=False, handlelength=1.4,
               columnspacing=1.0, handletextpad=0.45,
               bbox_to_anchor=(0.5, max(y - 0.05, 0.0)))


def save(fig, stem, note=""):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf, png = OUTDIR / f"{stem}.pdf", OUTDIR / f"{stem}.png"
    # NOTE: no bbox_inches="tight" -- it rescales the figure and every type
    # size with it, which is how the submitted figures lost their legibility.
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf} and {png}{note}")


# ==========================================================================
# Fig. 1 -- ground-truth SCM, verified against the generated data
# ==========================================================================
def ols_check(spec, data_path):
    """Re-estimate each structural coefficient from the generated data."""
    if not Path(data_path).exists():
        return None
    df = pd.read_csv(data_path)
    parents = defaultdict(list)
    for _, r in spec.iterrows():
        parents[r["to"]].append(r["from"])

    est = {}
    for child, ps in parents.items():
        usable = [p for p in ps if p in df.columns]
        if child not in df.columns or not usable:
            continue
        X = np.column_stack([np.ones(len(df))] + [df[p].to_numpy(float)
                                                  for p in usable])
        y = df[child].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        for k, p in enumerate(usable, start=1):
            est[(p, child)] = float(beta[k])
    return est


def fig1(args):
    spec = need_csv(PATHS["scm_spec"],
                    "columns: from,to,weight[,planted]  (exported from your "
                    "data-generating script)")
    for col in ("from", "to", "weight"):
        if col not in spec.columns:
            raise SystemExit(f"{PATHS['scm_spec']}: missing column {col!r}")
    if "planted" not in spec.columns:
        spec["planted"] = 0

    est = ols_check(spec, PATHS["synth_biased"])
    if est is None:
        print(f"  ! {PATHS['synth_biased']} not found -- skipping the "
              f"spec-vs-data check")
    else:
        bad = []
        for _, r in spec.iterrows():
            e = est.get((r["from"], r["to"]))
            if e is not None and abs(e - float(r["weight"])) > OLS_TOLERANCE:
                bad.append((r["from"], r["to"], float(r["weight"]), e))
        print(f"  spec-vs-data OLS check on {PATHS['synth_biased']}: "
              f"{len(spec) - len(bad)}/{len(spec)} edges agree within "
              f"{OLS_TOLERANCE}")
        for s, d, w, e in bad:
            print(f"    ! {s}->{d}: spec {w:+.4f} vs OLS {e:+.4f} "
                  f"(diff {abs(e-w):.4f})")
        if bad:
            print("    the figure prints the SPEC values; resolve this before "
                  "submitting")

    n_panels = args.synth_panels
    def panel_edges(include_planted):
        rows = spec if include_planted else spec[spec["planted"] == 0]
        return [(r["from"], r["to"], "directed") for _, r in rows.iterrows()], \
               {(r["from"], r["to"]): f"{float(r['weight']):.2f}"
                for _, r in rows.iterrows()}

    width = COL_W if n_panels == 1 else PAGE_W              #this code seems backwards, shouldn't be width = PAGE_W if n_panels == 1 else COL_W?
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(width, 2.7 if n_panels == 1 else 2.9))
    axes = np.atleast_1d(axes)

    nodes = sorted(set(spec["from"]) | set(spec["to"]))
    beta_planted = spec.loc[spec["planted"] == 1, "weight"]
    beta_txt = (f"$\\beta$={float(beta_planted.iloc[0]):.2f}"
                if len(beta_planted) else "planted")

    edges_a, labels_a = panel_edges(True)
    panel_aspect = (width / n_panels) / 2.05
    draw_dag(axes[0], nodes, edges_a, labels=labels_a,
             flags=FLAG_EDGES["fig1"],
             title=f"Biased SCM (Dataset A, {beta_txt})" if n_panels == 2 else None,
             node_w=0.88, node_h=0.46, x_gap=1.32, aspect=panel_aspect)

    if n_panels == 2:
        edges_b, labels_b = panel_edges(False)
        draw_dag(axes[1], nodes, edges_b, labels=labels_b, flags=(),
                 title="Unbiased SCM (Dataset B, $\\beta$=0.00)",
                 node_w=0.88, node_h=0.46, x_gap=1.32, aspect=panel_aspect)

    roles_used = []
    for v in nodes:
        r = NODE_ROLES.get(v, DEFAULT_ROLE)
        if r not in roles_used:
            roles_used.append(r)
    from matplotlib.lines import Line2D
    role_legend(fig, roles_used, y=0.11, extra=[
        Line2D([0], [0], color=FLAG_C, lw=1.3, ls=(0, (3.2, 2.0)),
               label="Planted direct bias")])
    fig.subplots_adjust(left=0.01, right=0.99,
                        top=0.92 if n_panels == 2 else 0.99,
                        bottom=0.30 if n_panels == 1 else 0.20, wspace=0.02)
    save(fig, "fig1_ground_truth_dag",
         f"  ({n_panels} panel{'s' if n_panels == 2 else ''}, "
         f"{len(spec)} edges from {PATHS['scm_spec']})")


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
    df = need_csv(PATHS["sensitivity"],
                  "columns: algorithm,n,beta,detection_rate,mean_shd,std_shd")
    need = {"algorithm", "n", "beta", "detection_rate", "mean_shd", "std_shd"}
    if need - set(df.columns):
        raise SystemExit(f"{PATHS['sensitivity']}: missing "
                         f"{sorted(need - set(df.columns))}")

    algorithms = [a for a in ["PC", "FCI", "GES", "GRaSP",
                              "ICA-LiNGAM", "DirectLiNGAM"]
                  if a in set(df.algorithm)]
    sizes = sorted(df.n.unique())
    betas = sorted(df.beta.unique())

    fig, axes = plt.subplots(len(sizes), len(algorithms),
                             figsize=(PAGE_W, 4.6), sharex=True, sharey=True,
                             squeeze=False)
    shd_max = max(4.0, float(df.mean_shd.max()) * 1.05)
    rows_out = []

    for i, n in enumerate(sizes):
        for j, alg in enumerate(algorithms):
            ax = axes[i][j]
            cell = df[(df.algorithm == alg) & (df.n == n)].set_index("beta")
            rates, means, stds = [], [], []
            for b in betas:
                if b not in cell.index:
                    raise SystemExit(f"{PATHS['sensitivity']}: no row for "
                                     f"algorithm={alg}, n={n}, beta={b}")
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
                          edgecolor="#1B1B1B", linewidth=0.3, zorder=3)
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

            if ANNOTATE_SHD:
                for xi, (r, m, sd) in enumerate(zip(rates, means, stds)):
                    ax.text(xi, min(r + 0.04, 0.72), fmt_shd(m, sd),
                            ha="center", va="bottom", rotation=90,
                            fontsize=5.6, color="#222222", zorder=5)

            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks(x)
            ax.set_xticklabels([f"{b:.2f}" for b in betas], fontsize=FS_TICK,
                               rotation=90)
            ax.tick_params(axis="y", labelsize=FS_TICK, length=1.5, pad=1)
            ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
                ax2.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_linewidth(0.4)
            if i == 0:
                ax.set_title(alg, fontsize=FS_SUB + 0.9, fontweight="bold", pad=4)
            if j == 0:
                ax.set_ylabel(f"$n$={n:,}\ndetection rate", fontsize=FS_AXIS,
                              labelpad=2)
            if i == len(sizes) - 1:
                ax.set_xlabel(r"$\beta$", fontsize=FS_AXIS, labelpad=1)

    fig.suptitle(r"Sensitivity analysis: Race$\rightarrow$Loan detection rate "
                 r"and SHD", fontsize=FS_TITLE, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "bar height and fill = detection rate; red line = mean SHD "
             "(right axis); red outline = main setting " r"$\beta$ = 0.15",
             ha="center", fontsize=FS_EDGE, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.945, top=0.895, bottom=0.105,
                        wspace=0.16, hspace=0.22)
    save(fig, "fig2_sensitivity_grid", f"  ({len(rows_out)} cells)")
    pd.DataFrame(rows_out).to_csv(OUTDIR / "fig2_shd_table.csv", index=False)
    print(f"  wrote {OUTDIR/'fig2_shd_table.csv'}")


# ==========================================================================
# Fig. 3 -- DirectLiNGAM on the synthetic biased dataset
# ==========================================================================
def fig3(args):
    m = need_matrix(PATHS["synth_dlingam_B"],
                    "square LiNGAM adjacency, B[i][j] = coefficient on j -> i")
    names = need_names(PATHS["synth_varnames"], m.shape[0])
    raw = decode_lingam(m, names)
    edges = [(s, d, k) for s, d, k, _ in raw]
    weights = {(s, d): w for s, d, _, w in raw}

    labels = {}
    for pair in LABEL_EDGES["fig3"]:
        if pair in weights:
            labels[pair] = f"{weights[pair]:+.4f}"

    nodes = sorted(set(names))
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    draw_dag(ax, nodes, edges, labels=labels, flags=FLAG_EDGES["fig3"],
             node_w=0.88, node_h=0.46, x_gap=1.32, aspect=COL_W / 2.05)

    roles_used = []
    for v in nodes:
        r = NODE_ROLES.get(v, DEFAULT_ROLE)
        if r not in roles_used:
            roles_used.append(r)
    from matplotlib.lines import Line2D
    role_legend(fig, roles_used, y=0.10, extra=[
        Line2D([0], [0], color=FLAG_C, lw=1.3, ls=(0, (3.2, 2.0)),
               label="Detected bias path")])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.29)
    save(fig, "fig3_synth_directlingam", f"  ({len(edges)} edges)")


# ==========================================================================
# Fig. 4 -- algorithm comparison on COMPAS
# ==========================================================================
def load_compas_panel(alg):
    path, kind = COMPAS_GRAPHS[alg]
    m = need_matrix(path, "square causal-learn graph matrix or LiNGAM adjacency")
    names = need_names(PATHS["compas_varnames"], m.shape[0])
    if kind == "lingam":
        raw = decode_lingam(m, names)
        return names, [(s, d, k) for s, d, k, _ in raw], \
               {(s, d): w for s, d, _, w in raw}
    return names, decode_causal_learn(m, names), {}


def fig4(args):
    panels = [a for a in COMPAS_PANELS if a in COMPAS_GRAPHS]
    loaded = []
    for alg in panels:
        try:
            loaded.append((alg,) + load_compas_panel(alg))
        except MissingInput as exc:
            if args.strict:
                raise
            print(f"  ! skipping panel {alg}: {exc}")
    if not loaded:
        raise SystemExit("  no COMPAS panels available")

    fig, axes = plt.subplots(1, len(loaded), figsize=(PAGE_W, 2.5),
                             squeeze=False)
    all_nodes = []
    for ax, (alg, names, edges, weights) in zip(axes[0], loaded):
        labels = {}
        if alg == "DirectLiNGAM":
            for pair in LABEL_EDGES["fig5"]:
                if pair in weights:
                    labels[pair] = f"{weights[pair]:+.4f}"
        draw_dag(ax, sorted(set(names)), edges, labels=labels,
                 flags=FLAG_EDGES["fig4"], title=alg,
                 node_w=0.90, node_h=0.44, fs_node=FS_NODE - 0.7,
                 fs_edge=FS_EDGE - 0.4, x_gap=1.34,
                 aspect=(PAGE_W / max(len(loaded), 1)) / 2.0)
        all_nodes += list(names)

    roles_used = []
    for v in all_nodes:
        r = NODE_ROLES.get(v, DEFAULT_ROLE)
        if r not in roles_used:
            roles_used.append(r)
    from matplotlib.lines import Line2D
    role_legend(fig, roles_used, y=0.10, extra=[
        Line2D([0], [0], color=MUTED_C, lw=0.8, label="Recovered edge"),
        Line2D([0], [0], color=FLAG_C, lw=1.3, ls=(0, (3.2, 2.0)),
               label="Flagged for review")])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.17,
                        wspace=0.03)
    save(fig, "fig4_compas_algorithm_grid",
         f"  ({len(loaded)} panels: {', '.join(a for a, *_ in loaded)})")


# ==========================================================================
# Fig. 5 -- DirectLiNGAM on COMPAS
# ==========================================================================
def fig5(args):
    m = need_matrix(PATHS["compas_dlingam_B"],
                    "square LiNGAM adjacency, B[i][j] = coefficient on j -> i")
    names = need_names(PATHS["compas_varnames"], m.shape[0])
    raw = decode_lingam(m, names)
    edges = [(s, d, k) for s, d, k, _ in raw]
    weights = {(s, d): w for s, d, _, w in raw}

    labels = {}
    for pair in LABEL_EDGES["fig5"]:
        if pair in weights:
            labels[pair] = f"{weights[pair]:+.4f}"
        else:
            print(f"  ! {pair[0]}->{pair[1]} is not in "
                  f"{PATHS['compas_dlingam_B']}; nothing labelled for it")

    n_txt = ""
    if Path(PATHS["compas_frame"]).exists():
        n_txt = f" ($n$={len(pd.read_csv(PATHS['compas_frame'])):,})"

    nodes = sorted(set(names))
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    draw_dag(ax, nodes, edges, labels=labels, flags=FLAG_EDGES["fig5"],
             title=f"DirectLiNGAM on COMPAS{n_txt}" if n_txt else None,
             node_w=0.88, node_h=0.46, x_gap=1.32, aspect=COL_W / 2.15)

    roles_used = []
    for v in nodes:
        r = NODE_ROLES.get(v, DEFAULT_ROLE)
        if r not in roles_used:
            roles_used.append(r)
    from matplotlib.lines import Line2D
    role_legend(fig, roles_used, y=0.10, extra=[
        Line2D([0], [0], color=FLAG_C, lw=1.3, ls=(0, (3.2, 2.0)),
               label="Detected bias path")])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92 if n_txt else 0.99,
                        bottom=0.28)
    save(fig, "fig5_compas_directlingam",
         f"  ({len(edges)} edges, {len(labels)} labelled)")


# ==========================================================================
# driver
# ==========================================================================
BUILDERS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}

CHECKLIST = [
    ("fig1", "scm_spec",         "from,to,weight[,planted]"),
    ("fig1", "synth_biased",     "the generated Dataset A (verification only)"),
    ("fig1", "synth_unbiased",   "the generated Dataset B (2-panel mode only)"),
    ("fig2", "sensitivity",      "algorithm,n,beta,detection_rate,mean_shd,std_shd"),
    ("fig3", "synth_dlingam_B",  "square LiNGAM adjacency"),
    ("fig3", "synth_varnames",   "one variable name per line"),
    ("fig5", "compas_dlingam_B", "square LiNGAM adjacency"),
    ("fig5", "compas_varnames",  "one variable name per line"),
    ("fig5", "compas_frame",     "the analysis frame (for the reported n)"),
]


def do_check():
    print("input check\n" + "-" * 66)
    missing = 0
    for fig, key, schema in CHECKLIST:
        p = PATHS[key]
        ok = Path(p).exists()
        missing += not ok
        print(f"  [{'ok' if ok else '--'}] {fig}  {p}")
        if not ok:
            print(f"          expected: {schema}")
    print("\n  Fig. 4 panels:")
    for alg, (p, kind) in COMPAS_GRAPHS.items():
        mark = "ok" if Path(p).exists() else "--"
        star = " *" if alg in COMPAS_PANELS else "  "
        print(f"  [{mark}]{star}{alg:<14} {p}  ({kind})")
    print("\n  * = drawn in Fig. 4 (edit COMPAS_PANELS to change)")
    print(f"\n  {missing} missing input(s).")
    return missing


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate every figure for ICTAI paper 582.")
    ap.add_argument("--only", default="",
                    help="comma-separated subset, e.g. fig1,fig2")
    ap.add_argument("--synth-panels", type=int, default=SYNTH_PANELS,
                    choices=[1, 2],
                    help="Fig. 1: 1 = biased SCM only, 2 = biased + unbiased")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--strict", action="store_true",
                    help="a missing input is fatal instead of a skip")
    ap.add_argument("--check", action="store_true",
                    help="report which inputs are present, build nothing")
    args = ap.parse_args()

    globals()["OUTDIR"] = Path(args.outdir)

    if args.check:
        sys.exit(0 if do_check() == 0 else 1)

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
