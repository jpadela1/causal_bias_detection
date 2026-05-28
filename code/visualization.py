"""
visualization.py  —  Graphviz-based causal graph renderer
==========================================================

BACKEND CHANGE: matplotlib FancyArrowPatch → Graphviz dot
----------------------------------------------------------
The matplotlib implementation required manual computation of
Bézier arc midpoints to position edge labels.  That math is fragile and
produced labels that drifted away from their arrows on long curved edges.

Graphviz (the dot/neato layout engine) places edge labels ON their edges
automatically and correctly — it is designed for exactly this task.
``rankdir=LR`` enforces the left→right causal reading direction so
outcomes always appear on the right, protected attributes on the left.

PUBLIC API IS UNCHANGED:
    plot_discovery_result(result, title, flagged_edges, node_roles, ...)
    plot_grid(results, flagged_edges, node_roles, title, ...)
    compute_shared_pos(results, ...)    # no-op — Graphviz handles layout
    save_figure_dual_format(fig, ...)  # kept for backward compatibility
    DEFAULT_ROLES_LOAN
    DEFAULT_ROLES_COMPAS

REQUIREMENTS:
    pip install graphviz          # Python wrapper (already installed)
    Graphviz binaries             # system install of dot/neato
        Windows : https://graphviz.org/download/  (add to PATH)
        macOS   : brew install graphviz
        Linux   : sudo apt install graphviz

OUTPUT:
    Every call saves <save_path>.pdf  (vector, infinitely zoomable)
    and               <save_path>.png (300 dpi raster for quick preview).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Iterable, Optional, Tuple

# Graphviz Python wrapper
try:
    import graphviz as gv
    _GV_AVAILABLE = True
except ImportError:
    _GV_AVAILABLE = False

# matplotlib — used only for plot_grid panel composition
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from causal_discovery import DiscoveryResult


# =============================================================================
# COLOR PALETTE  (unchanged from previous version)
# =============================================================================

COLOR_DIRECTED   = "#333333"
COLOR_UNDIRECTED = "#555555"   # dark grey — visible in print (was #888888)
COLOR_BIDIRECTED = "#9B59B6"
COLOR_FLAGGED    = "#D62728"

NODE_ROLE_COLORS = {
    "protected": "#A6CEE3",
    "proxy":     "#FDB863",
    "mediator":  "#A6DBA0",
    "outcome":   "#FBB4AE",
    "covariate": "#CCCCCC",
}

ROLE_LABELS = {
    "protected": "Protected attribute",
    "proxy":     "Proxy variable",
    "mediator":  "Mediator",
    "outcome":   "Outcome",
    "covariate": "Covariate",
}

DEFAULT_ROLES_LOAN = {
    "Race":      "protected",
    "Gender":    "protected",
    "ZIP":       "proxy",
    "Education": "covariate",
    "Income":    "mediator",
    "CreditSc":  "mediator",
    "Loan":      "outcome",
}

DEFAULT_ROLES_COMPAS = {
    "Race":         "protected",
    "Sex":          "protected",
    "Age":          "covariate",
    "JuvFelony":    "mediator",
    "JuvMisd":      "mediator",
    "Priors":       "proxy",
    "ChargeDegree": "covariate",
    "Score":        "outcome",
    "Recidivism":   "outcome",
}

# Rank-pinned node sets for rankdir=LR layout
_SOURCE_NODES = {"Race", "Gender", "Sex"}
_SINK_NODES   = {"Loan", "Score", "Recidivism"}


# =============================================================================
# GRAPHVIZ AVAILABILITY CHECK
# =============================================================================

def _check_graphviz() -> None:
    if not _GV_AVAILABLE:
        raise RuntimeError(
            "The 'graphviz' Python package is not installed.\n"
            "Run:  pip install graphviz"
        )
    if shutil.which("dot") is None:
        raise RuntimeError(
            "The Graphviz 'dot' binary was not found on PATH.\n"
            "Install Graphviz:\n"
            "  Windows : https://graphviz.org/download/  (tick 'Add to PATH')\n"
            "  macOS   : brew install graphviz\n"
            "  Linux   : sudo apt install graphviz\n"
            "Then restart your terminal / IDE."
        )


# =============================================================================
# CORE DOT GRAPH BUILDER
# =============================================================================

def _build_dot(
    result: DiscoveryResult,
    title: str,
    flagged: set,
    roles: dict,
    show_coefficients: bool,
    coef_threshold: float,
    node_font_size: int = 16,
    edge_font_size: int = 20,
    node_width: str = "1.7",
    ranksep: str = "4.0",        # ← new; widen for individual DAGs JSP 2.5
    canvas_size: str = "14,9",   # ← new; aspect ratio of the rendered PNG
) -> "gv.Digraph":
    """
    Convert one DiscoveryResult into a graphviz.Digraph.

    Layout
    ------
    rankdir=LR         : causal direction reads left to right.
    rank=min           : pins Race/Gender/Sex to the leftmost column.
    rank=same (mid)    : dataset-aware intermediate rank groups spread
                         intermediate nodes across the available width.
    rank=max           : pins Loan/Score/Recidivism to the rightmost column.
    splines=curved     : smooth cubic Bezier curves.  Tighter and less
                         circuitous than splines=spline for long-distance
                         edges such as Race→Loan that skip many rank columns.
    size / ratio=fill  : tells Graphviz to use the full page area so nodes
                         are spread across the right side of the canvas,
                         not compressed toward the left.

    Edge labels
    -----------
    Graphviz places the label string at the midpoint of the spline
    curve automatically — no manual Bézier math needed.
    """
    # Title is shown by matplotlib, not embedded in the dot graph, so that
    # the PNG we hand to matplotlib does not already have a title baked in.
    dot = gv.Digraph(
        name=title,
        graph_attr=dict(
            rankdir  = "LR",
            splines  = "spline",  # proper B-spline routing — clean, professional
            nodesep  = "0.9",
            ranksep  = ranksep,
            pad      = "0.6",
            bgcolor  = "white",
            fontname = "Helvetica",
            size     = canvas_size,
            ratio    = "fill",
            forcelabels = "true",   # always render xlabels even if crowded
        ),
        node_attr=dict(
            shape     = "circle",
            style     = "filled",
            fontname  = "Helvetica-Bold",
            fontsize  = str(node_font_size),
            fixedsize = "true",
            width     = node_width,
            height    = node_width,
        ),
        edge_attr=dict(
            fontname  = "Helvetica",
            fontsize  = str(edge_font_size),
            fontcolor = "#444444",
            penwidth  = "2.2",     # thicker edges for print legibility (was 1.6)
        ),
    )

    vars_set = set(result.variables)

    # ── Nodes ────────────────────────────────────────────────────────────────
    source_nodes, sink_nodes = [], []
    for v in result.variables:
        role  = roles.get(v, "covariate")
        fill  = NODE_ROLE_COLORS.get(role, "#CCCCCC")
        dot.node(v, label=v, fillcolor=fill, color="black", penwidth="2.2")
        if v in _SOURCE_NODES:
            source_nodes.append(v)
        if v in _SINK_NODES:
            sink_nodes.append(v)

    # ── Source rank (leftmost column) ─────────────────────────────────────
    if source_nodes:
        with dot.subgraph() as s:
            s.attr(rank="min")
            for v in source_nodes:
                s.node(v)

    # ── Sink rank (rightmost column) ──────────────────────────────────────
    if sink_nodes:
        with dot.subgraph() as s:
            s.attr(rank="max")
            for v in sink_nodes:
                s.node(v)

    # ── Dataset-aware intermediate rank groups ────────────────────────────
    # These force intermediate nodes into distinct rank columns so they
    # spread across the full canvas width rather than bunching together.
    _COMPAS_VARS = {"Race","Sex","Age","JuvFelony","JuvMisd",
                    "Priors","ChargeDegree","Score","Recidivism"}
    _LOAN_VARS   = {"Race","Gender","Education","ZIP","Income","CreditSc","Loan"}

    if vars_set.issubset(_COMPAS_VARS):
        # COMPAS: 5 rank columns
        #   min  → Race, Sex
        #   col2 → Age
        #   col3 → JuvFelony, JuvMisd
        #   col4 → Priors, ChargeDegree
        #   max  → Score, Recidivism
        for group in [["Age"], ["JuvFelony", "JuvMisd"], ["Priors", "ChargeDegree"]]:
            present = [v for v in group if v in vars_set]
            if present:
                with dot.subgraph() as s:
                    s.attr(rank="same")
                    for v in present:
                        s.node(v)

    elif vars_set.issubset(_LOAN_VARS):
        # Loan: 5 rank columns
        #   min  → Race, Gender
        #   col2 → Education
        #   col3 → ZIP, Income
        #   col4 → CreditSc
        #   max  → Loan
        for group in [["Education"], ["ZIP", "Income"], ["CreditSc"]]:
            present = [v for v in group if v in vars_set]
            if present:
                with dot.subgraph() as s:
                    s.attr(rank="same")
                    for v in present:
                        s.node(v)

    # ── Directed edges ────────────────────────────────────────────────────────
    for src, dst in result.directed_edges:
        is_fl = (src, dst) in flagged
        label = ""
        if show_coefficients and result.coef_matrix is not None:
            coef = result.get_coefficient(src, dst)
            if coef is not None and abs(coef) >= coef_threshold:
                label = f"{coef:+.4f}"
        if is_fl:
            dot.edge(src, dst,
                     xlabel     = label,      # xlabel works with splines=curved
                     color      = COLOR_FLAGGED,
                     style      = "dashed",
                     penwidth   = "2.8",
                     arrowsize  = "1.3",
                     fontcolor  = COLOR_FLAGGED,
                     fontname   = "Helvetica-Bold",
                     fontsize   = str(edge_font_size + 1),
                     constraint = "false",
                     weight     = "0.5")
        else:
            dot.edge(src, dst,
                     xlabel   = label,      # xlabel works with splines=curved
                     color    = COLOR_DIRECTED,
                     style    = "solid",
                     fontcolor= "#444444")

    # ── Undirected edges ──────────────────────────────────────────────────────
    # Dotted lines must be thick and dark enough to survive PDF→print scaling.
    # #555555 (dark grey) is far more visible than #888888 in print.
    # penwidth=3.0 ensures the dots are large enough to see at paper size.
    for src, dst in result.undirected_edges:
        dot.edge(src, dst,
                 dir     = "none",
                 style   = "dotted",
                 color   = "#555555",   # darker than before (#888888)
                 penwidth= "3.0")       # much thicker (was 1.3)

    # ── Bidirected edges (latent confounder) ──────────────────────────────────
    for src, dst in result.bidirected_edges:
        dot.edge(src, dst,
                 dir      = "both",
                 style    = "solid",
                 color    = COLOR_BIDIRECTED,
                 penwidth = "2.8",     # thicker (was 2.0)
                 arrowsize= "1.3")

    return dot


# =============================================================================
# LEGEND  (embedded as a Graphviz cluster)
# =============================================================================

def _add_legend_cluster(
    dot: "gv.Digraph",
    roles_present: list[str],
    has_bidirected: bool,
    has_flagged: bool,
) -> None:
    """
    Embed a legend inside the Graphviz graph as a cluster subgraph.

    Each row is a pair: a tiny dummy arrow node + a text-label node,
    connected by a styled invisible edge that acts as the legend icon.
    Node roles are shown as filled rectangles.
    """
    with dot.subgraph(name="cluster_legend") as leg:
        leg.attr(
            label     = "Legend",
            style     = "rounded,filled",
            fillcolor = "#f5f5f5",
            color     = "#bbbbbb",
            fontname  = "Helvetica",
            fontsize  = "10",
            penwidth  = "1.0",
            margin    = "12",
            rank      = "sink",
        )
        leg.attr("node",
                 shape    = "none",
                 margin   = "0",
                 fontname = "Helvetica",
                 fontsize = "9",
                 width    = "0.1",
                 height   = "0.1",
                 style    = "invis")

        prev = None

        def _leg_row(nid_src, nid_dst, text, color, style, direction, pw):
            nonlocal prev
            leg.node(nid_src, label="", style="invis", width="0.1", height="0.1")
            leg.node(nid_dst,
                     label    = f'<<FONT FACE="Helvetica" POINT-SIZE="9">'
                                f'{text}</FONT>>',
                     style    = "invis")
            leg.edge(nid_src, nid_dst,
                     color    = color,
                     style    = style,
                     dir      = direction,
                     penwidth = pw,
                     arrowsize= "0.6",
                     minlen   = "1")
            if prev:
                leg.edge(prev, nid_src, style="invis", weight="10")
            prev = nid_dst

        _leg_row("ld_s", "ld_t", "Directed (i → j)",
                 COLOR_DIRECTED,   "solid",  "forward", "1.4")
        _leg_row("lu_s", "lu_t", "Undirected (i — j)",
                 COLOR_UNDIRECTED, "dotted", "none",    "1.1")
        if has_bidirected:
            _leg_row("lb_s", "lb_t", "Bidirected (latent confounder)",
                     COLOR_BIDIRECTED, "solid", "both", "1.8")
        if has_flagged:
            _leg_row("lf_s", "lf_t", "Flagged edge (for review)",
                     COLOR_FLAGGED, "dashed", "forward", "2.4")

        for role in roles_present:
            fill  = NODE_ROLE_COLORS.get(role, "#CCCCCC")
            text  = ROLE_LABELS.get(role, role)
            nid   = f"lr_{role}"
            leg.node(nid,
                     label    = f'<<TABLE BORDER="1" CELLBORDER="0" '
                                f'BGCOLOR="{fill}" STYLE="ROUNDED">'
                                f'<TR><TD ALIGN="LEFT">&nbsp;{text}&nbsp;'
                                f'</TD></TR></TABLE>>',
                     style    = "invis")
            if prev:
                leg.edge(prev, nid, style="invis", weight="10")
            prev = nid


# =============================================================================
# RENDER / SAVE HELPERS
# =============================================================================

def _render_final(fig, save_path: str, dpi: int = 200) -> None:
    """Save a matplotlib figure (Graphviz+legend composite) as PDF and PNG."""
    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  saved: {base}.pdf")
    print(f"  saved: {base}.png")


def _render_dot(dot: "gv.Digraph", save_path: str) -> None:
    """Save as PDF (vector) and PNG (raster 200 dpi)."""
    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    pdf_path = base + ".pdf"
    dot.render(outfile=pdf_path, format="pdf", cleanup=True)
    print(f"  saved: {pdf_path}")

    png_path = base + ".png"
    dot_png  = dot.copy()
    dot_png.attr(dpi="200")
    dot_png.render(outfile=png_path, format="png", cleanup=True)
    print(f"  saved: {png_path}")


def _render_dot_to_png_only(dot: "gv.Digraph", out_png: str) -> None:
    """Render to a single PNG — used internally by plot_grid."""
    dot.attr(dpi="150")
    dot.render(outfile=out_png, format="png", cleanup=True)


def save_figure_dual_format(fig, save_path: str, dpi: int = 300) -> None:
    """
    Backward-compatibility shim for matplotlib Figure objects.

    Non-graph matplotlib figures (sensitivity heatmap, ATE bar chart, etc.)
    still use this.  DAG figures now use _render_dot() instead.
    """
    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    fig.savefig(base + ".png", dpi=dpi,  bbox_inches="tight")
    fig.savefig(base + ".pdf",           bbox_inches="tight")
    print(f"  saved: {base}.png")
    print(f"  saved: {base}.pdf")


# =============================================================================
# BACKWARD-COMPAT NO-OPS
# =============================================================================

def compute_shared_pos(
    results: dict,
    layout: str = "fixed",
    seed: int = 7,
) -> dict:
    """No-op — Graphviz handles layout automatically. Returns empty dict."""
    return {}


def _print_edge_summary(result: DiscoveryResult) -> None:
    n_dir  = len(result.directed_edges)
    n_und  = len(result.undirected_edges)
    n_bi   = len(result.bidirected_edges)
    total  = n_dir + n_und + n_bi
    if total == 0:
        print(f"  [{result.algorithm}] 0 edges discovered")
        return
    pct = 100.0 * n_dir / total
    print(f"  [{result.algorithm}] {n_dir} directed, {n_und} undirected, "
          f"{n_bi} bidirected  ({pct:.0f}% oriented)")


# =============================================================================
# PUBLIC PLOT FUNCTIONS
# =============================================================================

def plot_discovery_result(
    result: DiscoveryResult,
    title: Optional[str] = None,
    flagged_edges: Optional[Iterable[Tuple[str, str]]] = None,
    node_roles: Optional[dict] = None,
    show_coefficients: bool = True,
    coef_threshold: float = 0.05,
    layout: str = "fixed",       # accepted for API compat, ignored
    pos: Optional[dict] = None,  # accepted for API compat, ignored
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 9),
    show_legend: bool = True,
):
    """
    Render one DiscoveryResult as a publication-quality causal DAG.

    Strategy
    --------
    1. Graphviz dot renders the graph to a temp PNG.
       - rankdir=LR  : outcomes on the right, protected attrs on the left.
       - Dataset-aware rank groups spread intermediate nodes evenly.
       - Edge labels (β coefficients) placed ON the edge automatically.
       - size/ratio=fill uses the full canvas so space on the right is used.

    2. matplotlib loads the PNG and composes the final figure:
       - Graph image occupies the top ~85% of the figure.
       - Legend occupies the bottom ~12%, clearly separated from the graph.
       - Title is set as a matplotlib suptitle (not baked into the PNG).

    This hybrid approach gives Graphviz-quality edge-label placement AND
    a clean, well-positioned legend that cannot drift into the graph area.
    """
    _check_graphviz()

    flagged = set(flagged_edges or [])
    roles   = node_roles or {v: "covariate" for v in result.variables}
    heading = title or result.algorithm

    _print_edge_summary(result)

    dot = _build_dot(
        result=result,
        title=heading,
        flagged=flagged,
        roles=roles,
        show_coefficients=show_coefficients,
        coef_threshold=coef_threshold,
        ranksep="3.5",  # ← new; widens the individual DAGs
        canvas_size="16,9",  # ← new; matches figsize aspect
    )

    # ── Render Graphviz → temp PNG ────────────────────────────────────────────
    tmpdir  = tempfile.mkdtemp(prefix="causal_plot_")
    try:
        tmp_png = os.path.join(tmpdir, "graph.png")
        dot.attr(dpi="200")
        dot.render(outfile=tmp_png, format="png", cleanup=True)

        # ── Compose in matplotlib ─────────────────────────────────────────────
        legend_frac = 0.12 if show_legend else 0.0
        title_frac = 0.10  # reserve top 10% for the title
        fig = plt.figure(figsize=figsize, facecolor="white")

        # Graph image — fills everything above the legend strip
        ax_g = fig.add_axes([0.0, legend_frac, 1.0, 1.0 - legend_frac - title_frac])
        if os.path.exists(tmp_png):
            ax_g.imshow(plt.imread(tmp_png), interpolation="lanczos")
        ax_g.axis("off")

        # Title — set as a matplotlib suptitle so it appears above the image
        fig.suptitle(heading, fontsize=13, fontweight="bold", y=0.96)

        # Legend — horizontal strip at the bottom of the figure
        if show_legend:
            roles_present = sorted({roles.get(v, "covariate")
                                     for v in result.variables})
            has_fl  = bool(flagged & {(s, d) for s, d in result.directed_edges})
            has_bi  = bool(result.bidirected_edges)
            has_und = bool(result.undirected_edges)

            edge_handles = [
                Line2D([0],[0], color=COLOR_DIRECTED,   lw=1.6,
                       marker=">", markersize=8, label="Directed (i → j)"),
            ]
            if has_und:
                edge_handles.append(
                    Line2D([0],[0], color=COLOR_UNDIRECTED, lw=1.0,
                           linestyle=":", label="Undirected (i — j)"))
            if has_bi:
                edge_handles.append(
                    Line2D([0],[0], color=COLOR_BIDIRECTED, lw=1.8,
                           marker=">", markersize=8,
                           label="Bidirected (latent confounder)"))
            if has_fl:
                edge_handles.append(
                    Line2D([0],[0], color=COLOR_FLAGGED, lw=2.2,
                           linestyle="--", marker=">", markersize=8,
                           label="Flagged edge (for review)"))
            node_handles = [
                mpatches.Patch(facecolor=NODE_ROLE_COLORS[r], edgecolor="black",
                               label=ROLE_LABELS[r])
                for r in roles_present
            ]
            all_handles = edge_handles + node_handles

            # Dedicated axes for the legend strip
            ax_l = fig.add_axes([0.0, 0.0, 1.0, legend_frac])
            ax_l.axis("off")
            ax_l.legend(
                handles=all_handles,
                loc="center",
                ncol=min(len(all_handles), 6),
                frameon=True, fancybox=True, framealpha=0.95,
                edgecolor="#cccccc", fontsize=9,
                borderaxespad=0.3,
            )

        if save_path:
            _render_final(fig, save_path)

        return fig

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def plot_grid(
    results: dict,
    flagged_edges: Optional[Iterable[Tuple[str, str]]] = None,
    node_roles: Optional[dict] = None,
    title: str = "Discovered DAGs",
    save_path: Optional[str] = None,
    layout: str = "fixed",
    pos: Optional[dict] = None,
    figsize_per_panel: Tuple[float, float] = (7.5, 5.0),
    panel_titles: Optional[dict] = None,     # ← new
):
    """
    Render all algorithm results in a 3-column grid.

    Each panel is rendered by Graphviz independently (so labels are always
    on their edges), then the PNG outputs are tiled into a single matplotlib
    figure for the overview grid.

    Saves <save_path>.pdf (composed grid) and <save_path>.png.
    """
    _check_graphviz()

    valid = [(k, v) for k, v in results.items() if v is not None]
    n     = len(valid)
    if n == 0:
        print("No valid results to plot.")
        return None

    flagged = set(flagged_edges or [])
    roles   = node_roles or {}

    tmpdir = tempfile.mkdtemp(prefix="causal_grid_")
    try:
        # ── Render each panel independently ──────────────────────────────────
        panel_pngs: list[tuple[str, str]] = []
        for alg_name, res in valid:
            dot = _build_dot(
                result            = res,
                title             = alg_name,
                flagged           = flagged,
                roles             = roles,
                show_coefficients = True,
                coef_threshold    = 0.05,
                node_font_size    = 13,    # was 11
                edge_font_size    = 11,    # was 8
                node_width        = "1.15", # was 0.95
            )
            # No per-panel legend — a shared legend sits below the grid
            out_png = os.path.join(tmpdir, f"{alg_name}.png")
            _render_dot_to_png_only(dot, out_png)
            panel_pngs.append((alg_name, out_png))

        # ── Compose panels into matplotlib figure ─────────────────────────────
        cols  = 3
        rows  = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
        )
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, (alg_name, png_path) in zip(axes, panel_pngs):
            if os.path.exists(png_path):
                img = plt.imread(png_path)
                ax.imshow(img, interpolation="lanczos")
            panel_label = (panel_titles or {}).get(alg_name, alg_name)
            ax.set_title(panel_label, fontsize=12, fontweight="bold", pad=6)
            ax.axis("off")

        for ax in axes[n:]:
            ax.set_axis_off()

        # ── Shared legend ─────────────────────────────────────────────────────
        roles_present = sorted({roles.get(v, "covariate")
                                 for _, res in valid
                                 for v in res.variables})
        edge_handles = [
            Line2D([0],[0], color=COLOR_DIRECTED,   lw=1.6,
                   marker=">", markersize=8, label="Directed (i → j)"),
            Line2D([0],[0], color=COLOR_UNDIRECTED, lw=1.0, linestyle=":",
                   label="Undirected (i — j)"),
            Line2D([0],[0], color=COLOR_BIDIRECTED, lw=1.8,
                   marker=">", markersize=8, label="Bidirected (latent confounder)"),
            Line2D([0],[0], color=COLOR_FLAGGED,    lw=2.2, linestyle="--",
                   marker=">", markersize=8, label="Flagged edge (for review)"),
        ]
        node_handles = [
            mpatches.Patch(facecolor=NODE_ROLE_COLORS[r], edgecolor="black",
                           label=ROLE_LABELS[r])
            for r in roles_present
        ]
        all_handles = edge_handles + node_handles

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.subplots_adjust(
            top=0.93, bottom=0.10,
            left=0.01, right=0.99,
            hspace=0.12, wspace=0.03,
        )
        fig.legend(
            handles=all_handles,
            loc="lower center",
            ncol=min(len(all_handles), 5),
            frameon=True, fancybox=True, framealpha=0.95,
            edgecolor="#cccccc", fontsize=9,
            bbox_to_anchor=(0.5, 0.01),
        )

        if save_path:
            save_figure_dual_format(fig, save_path)

        return fig

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
