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
    plot_edge_list(edges, title, node_roles, flagged_edges, proxy_edges, ...)
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

import math
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
#import matplotlib
#matplotlib.use("Agg")
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
COLOR_PROXY      = "#D9820A"   # proxy-discrimination pathway (ground-truth DAG)
COLOR_LATENT_EDGE = "#888888"  # edges originating at an unobserved node

NODE_ROLE_COLORS = {
    "protected": "#A6CEE3",
    "proxy":     "#FDB863",
    "mediator":  "#A6DBA0",
    "outcome":   "#FBB4AE",
    "covariate": "#CCCCCC",
    "latent":    "#FFFFFF",
}

# Shape is a REDUNDANT encoding of role, so the figure still reads when the
# journal prints it in greyscale.  The five fills above have nearly identical
# luminance (#CCCCCC and #A6DBA0 both land near L*=80), so colour alone does
# not survive a black-and-white printer or a colour-blind reader.
NODE_ROLE_SHAPES = {
    "protected": "octagon",
    "proxy":     "hexagon",
    "mediator":  "ellipse",
    "outcome":   "box",
    "covariate": "ellipse",
    "latent":    "diamond",
}

# Extra outline for roles whose shape would otherwise collide (covariate and
# mediator are both ellipses; the double outline separates them in greyscale).
NODE_ROLE_PERIPHERIES = {
    "covariate": "2",
}

# Per-role node style.  "latent" carries a dashed border on top of its own
# shape: an unobserved variable is conventionally drawn dashed, and the extra
# cue costs nothing in greyscale (the white fill alone would be ambiguous
# against the page).  Roles absent here use the graph-level "filled".
NODE_ROLE_STYLES = {
    "latent": "filled,dashed",
}

ROLE_LABELS = {
    "protected": "Protected (octagon)",
    "proxy":     "Proxy (hexagon)",
    "mediator":  "Mediator (ellipse)",
    "outcome":   "Outcome (rectangle)",
    "covariate": "Covariate (double ellipse)",
    "latent":    "Latent, unobserved (dashed diamond)",
}

# =============================================================================
# LAYOUT PRESETS  —  the key to legible output
# =============================================================================
# The renderer deliberately does NOT set Graphviz's `size` or `ratio`
# attributes.  `size` is a *maximum*: whenever the natural layout is larger,
# Graphviz writes a uniform scale factor into the output, shrinking every
# font and stroke.  That is what made earlier versions unreadable — raising
# `node_width`/`fontsize` grew the natural layout, which lowered the scale
# factor, which cancelled the change out exactly.
#
# Instead: pick the preset whose *natural* width matches the width the figure
# will occupy on paper, then \includegraphics it at 1:1 (or near it).  Then a
# 12 pt font in this file is a 12 pt font on the printed page.
#
# Natural width  ~=  n_ranks*node_width + (n_ranks-1)*ranksep + edge-label space
# Both studies lay out in 5 rank columns.
LAYOUT_PRESETS = {
    # ~7.5 in wide: two-column paper, full-width float (\begin{figure*}).
    # Include with \includegraphics[width=\textwidth]{...} -> scale ~0.95.
    "fullwidth": dict(node_font_size=12, edge_font_size=9,
                      node_width="0.95", node_height="0.52",
                      ranksep="0.55", nodesep="0.30",
                      edge_penwidth="1.4", node_penwidth="1.0"),
    # ~3.6 in wide: single column.  Tight — consider show_coefficients=False.
    "column":    dict(node_font_size=9, edge_font_size=7,
                      node_width="0.62", node_height="0.38",
                      ranksep="0.30", nodesep="0.16",
                      edge_penwidth="1.0", node_penwidth="0.8"),
    # Panels inside plot_grid: each occupies ~1/3 of a full-width float, so
    # the type must be proportionally larger to survive that reduction.
    "panel":     dict(node_font_size=16, edge_font_size=12,
                      node_width="1.10", node_height="0.62",
                      ranksep="0.60", nodesep="0.28",
                      edge_penwidth="1.9", node_penwidth="1.4"),
}

DEFAULT_PRESET = "fullwidth"   # <- switch to "column" for single-column figures


# =============================================================================
# TITLE SANITIZING
# =============================================================================
# Titles were written for matplotlib's mathtext ($\hat{\beta}$).  Graphviz has
# no mathtext, so those would print literally as "$\hat{\beta}$".  Map the few
# constructs actually used to Unicode instead.
_MATHTEXT_REPLACEMENTS = (
    (r"$\hat{\beta}$", "β̂"),   # beta with combining circumflex
    (r"\hat{\beta}",   "β̂"),
    (r"$\beta$",       "β"),
    (r"\beta",         "β"),
)


def _plain_title(text: str) -> str:
    """Convert a matplotlib-mathtext title into a Graphviz-safe label.

    Returns a string using Graphviz's ``\\n`` line-break escape, with any
    leftover TeX punctuation removed.
    """
    out = text
    for tex, uni in _MATHTEXT_REPLACEMENTS:
        out = out.replace(tex, uni)
    # Drop any remaining TeX scaffolding, then re-introduce line breaks using
    # Graphviz's own escape (order matters: strip backslashes first).
    out = out.replace("$", "").replace("\\", "")
    out = out.replace('"', "'")
    return out.replace("\n", "\\n")


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

# Rank-pinned node sets for rankdir=LR layout.  SES is exogenous in the
# ground-truth SCM and never appears in a discovery result (it is the
# unobserved confounder), so pinning it left affects only the ground-truth DAG.
_SOURCE_NODES = {"Race", "Gender", "Sex", "SES"}
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
    coef_decimals: int = 4,
    preset: str = DEFAULT_PRESET,
    show_title: bool = True,
    # Per-call overrides.  None means "take the value from `preset`".
    node_font_size: Optional[int] = None,
    edge_font_size: Optional[int] = None,
    node_width: Optional[str] = None,
    ranksep: Optional[str] = None,
    nodesep: Optional[str] = None,
    canvas_size: Optional[str] = None,   # accepted for API compat; see note
    # Ground-truth extras.  Both default to "off", so a DiscoveryResult built
    # by any discovery algorithm takes exactly the path it did before.
    proxy: Optional[set] = None,
    edge_labels: Optional[dict] = None,
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

    NO `size`, NO `ratio`
    ---------------------
    ``canvas_size`` is accepted but ignored, on purpose.  Setting Graphviz's
    ``size`` caps the drawing and makes Graphviz emit a uniform scale factor,
    which shrinks every font and stroke in the output — the drawing is then
    reduced a *second* time when LaTeX fits it to the column.  Two successive
    reductions is what made these figures illegible.  Sizing is controlled
    instead by LAYOUT_PRESETS, which targets the printed width directly.

    Node shape
    ----------
    ``fixedsize=false`` lets each node grow to fit its label, with the
    preset's ``node_width``/``node_height`` acting as a *minimum* so nodes
    stay visually uniform.  Shape encodes role redundantly with fill colour
    so the figure survives greyscale printing.

    Edge labels
    -----------
    ``label`` (not ``xlabel``) is used so Graphviz reserves layout space for
    each coefficient on its spline.  ``xlabel`` floats free and, combined
    with ``forcelabels=true``, is explicitly permitted to overlap.
    """
    cfg = dict(LAYOUT_PRESETS[preset])
    if node_font_size is not None:
        cfg["node_font_size"] = node_font_size
    if edge_font_size is not None:
        cfg["edge_font_size"] = edge_font_size
    if node_width is not None:
        cfg["node_width"] = node_width
    if ranksep is not None:
        cfg["ranksep"] = ranksep
    if nodesep is not None:
        cfg["nodesep"] = nodesep

    graph_attr = dict(
        rankdir  = "LR",
        splines  = "spline",
        nodesep  = cfg["nodesep"],
        ranksep  = cfg["ranksep"],
        pad      = "0.12",
        bgcolor  = "white",
        fontname = "Helvetica-Bold",
        fontsize = str(cfg["node_font_size"] + 1),
        # `size` and `ratio` deliberately omitted — see docstring.
        # `forcelabels` deliberately omitted — it is what allows label overlap.
    )
    if show_title and title:
        graph_attr["label"]     = _plain_title(title)
        graph_attr["labelloc"]  = "t"
        graph_attr["labeljust"] = "c"

    dot = gv.Digraph(
        name="G",
        graph_attr=graph_attr,
        node_attr=dict(
            style     = "filled",
            fontname  = "Helvetica-Bold",
            fontsize  = str(cfg["node_font_size"]),
            fontcolor = "black",
            fixedsize = "false",       # label drives size; width is a minimum
            width     = cfg["node_width"],
            height    = cfg["node_height"],
            margin    = "0.06,0.035",
        ),
        edge_attr=dict(
            fontname  = "Helvetica",
            fontsize  = str(cfg["edge_font_size"]),
            fontcolor = "black",       # grey edge labels vanish first in print
            penwidth  = cfg["edge_penwidth"],
            arrowsize = "0.8",
        ),
    )

    vars_set = set(result.variables)

    # ── Nodes ────────────────────────────────────────────────────────────────
    source_nodes, sink_nodes = [], []
    for v in result.variables:
        role  = roles.get(v, "covariate")
        fill  = NODE_ROLE_COLORS.get(role, "#CCCCCC")
        shape = NODE_ROLE_SHAPES.get(role, "ellipse")
        attrs = dict(
            label     = v,
            shape     = shape,
            fillcolor = fill,
            color     = "black",
            penwidth  = cfg["node_penwidth"],
        )
        if role in NODE_ROLE_STYLES:
            attrs["style"] = NODE_ROLE_STYLES[role]
        if role in NODE_ROLE_PERIPHERIES:
            attrs["peripheries"] = NODE_ROLE_PERIPHERIES[role]
        dot.node(v, **attrs)
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
    # SES is present only in the ground-truth DAG; including it here keeps that
    # graph inside the Loan branch so it gets the same rank columns.
    _LOAN_VARS   = {"Race","Gender","Education","ZIP","Income","CreditSc",
                    "Loan","SES"}

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
    proxy_set = set(proxy or ())
    for src, dst in result.directed_edges:
        is_fl = (src, dst) in flagged
        label = ""
        if show_coefficients:
            if edge_labels is not None and (src, dst) in edge_labels:
                # Caller-supplied label (the ground-truth DAG passes the SCM's
                # own coefficient strings, e.g. "0.40" / "β=−0.15").
                label = str(edge_labels[(src, dst)])
            elif result.coef_matrix is not None:
                coef = result.get_coefficient(src, dst)
                if coef is not None and abs(coef) >= coef_threshold:
                    label = f"{coef:+.{coef_decimals}f}"

        # Proxy pathway and latent-source edges are ground-truth-only tiers:
        # `proxy` is empty for discovery results, and no discovery result
        # contains a node whose role is "latent", so neither branch fires there.
        is_proxy     = (src, dst) in proxy_set
        is_latent_src = roles.get(src) == "latent"

        if is_fl:
            # The one edge under test. Dash + colour + extra weight; the dash
            # pattern is reserved for this edge alone so it is unambiguous.
            dot.edge(src, dst,
                     label      = label,
                     color      = COLOR_FLAGGED,
                     style      = "dashed",
                     penwidth   = str(float(cfg["edge_penwidth"]) + 0.5),
                     arrowsize  = "0.9",
                     fontcolor  = COLOR_FLAGGED,
                     fontname   = "Helvetica-Bold",
                     fontsize   = str(cfg["edge_font_size"] + 1),
                     constraint = "false",
                     weight     = "0.5")
        elif is_proxy:
            # Proxy-discrimination pathway.  Distinguished from an ordinary
            # edge by STROKE WEIGHT, not colour, so the pathway still traces
            # in greyscale; the dash pattern stays reserved for the flagged
            # edge so the two tiers never collide in black and white.
            dot.edge(src, dst,
                     label     = label,
                     color     = COLOR_PROXY,
                     style     = "solid",
                     penwidth  = str(float(cfg["edge_penwidth"]) + 1.2),
                     arrowsize = "0.9",
                     fontcolor = COLOR_PROXY,
                     fontname  = "Helvetica-Bold")
        elif is_latent_src:
            # Edge out of an unobserved node: thin and grey, matching the
            # dashed node border so the whole latent channel recedes.
            dot.edge(src, dst,
                     label     = label,
                     color     = COLOR_LATENT_EDGE,
                     style     = "dashed",
                     penwidth  = str(max(float(cfg["edge_penwidth"]) - 0.4, 0.6)),
                     fontcolor = "#555555")
        else:
            dot.edge(src, dst,
                     label     = label,
                     color     = COLOR_DIRECTED,
                     style     = "solid",
                     fontcolor = "black")

    # ── Undirected edges ──────────────────────────────────────────────────────
    # Distinguished by the ABSENCE of an arrowhead, not by color or dash
    # pattern: dotted strokes disintegrate under reduction, and grey is the
    # first thing a printer loses.
    for src, dst in result.undirected_edges:
        dot.edge(src, dst,
                 dir      = "none",
                 style    = "solid",
                 color    = COLOR_UNDIRECTED,
                 penwidth = cfg["edge_penwidth"])

    # ── Bidirected edges (latent confounder) ──────────────────────────────────
    for src, dst in result.bidirected_edges:
        dot.edge(src, dst,
                 dir      = "both",
                 style    = "solid",
                 color    = COLOR_BIDIRECTED,
                 penwidth = str(float(cfg["edge_penwidth"]) + 0.3),
                 arrowsize= "0.9")

    return dot


# =============================================================================
# LEGEND  (embedded as a Graphviz cluster)
# =============================================================================

def _add_legend_cluster(
    dot: "gv.Digraph",
    roles_present: list[str],
    has_bidirected: bool,
    has_flagged: bool,
    has_undirected: bool = True,
    rank_with: Optional[list[str]] = None,
    has_proxy: bool = False,
    has_latent_edge: bool = False,
    flagged_label: str = "Flagged (for review)",
) -> None:
    """
    Embed a legend in the graph as a single HTML-table node.

    Why one node
    ------------
    The obvious encoding — a dummy node pair per row, joined by a styled edge
    that acts as the icon — cannot work under ``rankdir=LR``.  Each pair
    consumes two ranks, so the legend spreads across the drawing horizontally
    instead of stacking, and rank/cluster hints cannot pull it back: `rankdir`
    is a whole-graph attribute, and Graphviz gives no way to pin a cluster to
    the bottom edge.  One table node has exactly one rank to place, so the
    layout is predictable, and its rows stack the way a legend should.

    Edge styles are shown with coloured Unicode glyphs rather than real
    arrows, which is the trade for that predictability.  Role swatches are
    real filled cells, matching the node fills exactly.
    """
    rows: list[str] = []

    def _icon_row(glyph: str, color: str, text: str) -> None:
        rows.append(
            f'<TR>'
            f'<TD ALIGN="CENTER" WIDTH="18">'
            f'<FONT COLOR="{color}" POINT-SIZE="10"><B>{glyph}</B></FONT></TD>'
            f'<TD ALIGN="LEFT">{text}</TD>'
            f'</TR>'
        )

    _icon_row("&#8594;", COLOR_DIRECTED, "Directed (i &#8594; j)")
    if has_undirected:
        _icon_row("&#8212;", COLOR_UNDIRECTED, "Undirected (i &#8212; j)")
    if has_bidirected:
        _icon_row("&#8596;", COLOR_BIDIRECTED, "Bidirected (latent confounder)")
    if has_proxy:
        # Em dash + solid pointer reads as a heavy unbroken arrow, mirroring
        # the pathway's heavier stroke.  Both glyphs are already used by the
        # rows above, so they are known to exist in the legend font — the
        # dingbat arrows (U+279E and friends) are not, and render as tofu.
        _icon_row("&#8212;&#9658;", COLOR_PROXY,
                  "Proxy pathway (heavy stroke)")
    if has_flagged:
        # en-dashes read as a dashed stroke at this size
        _icon_row("&#8211;&#8211;&#9658;", COLOR_FLAGGED, flagged_label)
    if has_latent_edge:
        _icon_row("&#8211;&#8211;&#9658;", COLOR_LATENT_EDGE,
                  "Edge from latent node (thin dashed)")

    for role in roles_present:
        fill = NODE_ROLE_COLORS.get(role, "#CCCCCC")
        rows.append(
            f'<TR>'
            f'<TD BGCOLOR="{fill}" WIDTH="34" BORDER="1" COLOR="black"> </TD>'
            f'<TD ALIGN="LEFT">{ROLE_LABELS.get(role, role)}</TD>'
            f'</TR>'
        )

    label = (
        '<<TABLE BORDER="1" COLOR="#bbbbbb" CELLBORDER="0" CELLSPACING="1" '
        'CELLPADDING="1" BGCOLOR="white">'
       # '<TR><TD COLSPAN="2" ALIGN="CENTER"><B>Legend</B></TD></TR>'
        + "".join(rows) +
        '</TABLE>>'
    )
#    dot.node("_legend", label=label, shape="plaintext",
 #            fontname="Helvetica", fontsize="10", margin="0")

    dot.node("_legend", label=label, shape="plaintext",
             fontname="Helvetica", fontsize="8", margin="0")

    # Park the legend in the same rank (column, under rankdir=LR) as the
    # graph's source nodes, so it stacks with them instead of stretching
    # across the drawing.  It lands above them; forcing it below with a flat
    # invisible edge does work, but it also reorders the real source nodes
    # (Age gets dragged out of the column and the DAG distorts), so the
    # position is left to Graphviz.
    if rank_with:
        with dot.subgraph() as s:
            s.attr(rank="same")
            for v in rank_with:
                s.node(v)
            s.node("_legend")


# =============================================================================
# RENDER / SAVE HELPERS
# =============================================================================

def _render_final(fig, save_path: str, dpi: int = 300) -> None:
    """Save a matplotlib figure (Graphviz+legend composite) as PDF and PNG.  dpi: int = 200"""
    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    fig.savefig(base + ".pdf", facecolor="white")      #removed bbox_inches="tight"
    fig.savefig(base + ".png", dpi=dpi, facecolor="white") #removed bbox_inches="tight"
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
    dot_png.attr(dpi="300")         #dot_png.attr(dpi="200")
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
    fig.savefig(base + ".png", dpi=dpi)   #removed bbox_inches="tight"
    fig.savefig(base + ".pdf")       #removed bbox_inches="tight"
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
    coef_decimals: int = 4,      # match the precision used in the paper's tables
    layout: str = "fixed",       # accepted for API compat, ignored
    pos: Optional[dict] = None,  # accepted for API compat, ignored
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),     #figsize: Tuple[int, int] = (16, 9),
    show_legend: bool = True,
):
    """
    Render one DiscoveryResult as a publication-quality causal DAG.

    Strategy — Graphviz only, no matplotlib
    ---------------------------------------
    dot draws the graph, its title, and its legend in one pass, and
    ``_render_dot`` writes it straight to PDF and PNG:
       - rankdir=LR  : outcomes on the right, protected attrs on the left.
       - Dataset-aware rank groups spread intermediate nodes evenly.
       - Edge labels (β coefficients) placed ON the edge automatically.
       - Title is the graph's own label, so it scales with the drawing.
       - Legend is a cluster subgraph pinned below the graph.

    The earlier version composed this PNG into a matplotlib figure to add a
    suptitle and a legend strip.  That cost the PDF its vector text (the whole
    drawing arrived as a raster image), and it printed the title twice — once
    baked in by Graphviz, once drawn by matplotlib on top.  Everything
    matplotlib was doing here, dot already does natively.

    Returns the ``graphviz.Digraph`` (no caller uses the old Figure return).
    ``figsize`` is accepted for API compatibility and ignored: the drawing's
    natural size is set by LAYOUT_PRESETS, then scaled by \\includegraphics.
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
        coef_decimals=coef_decimals,
        show_title=True,
    )

    if show_legend:
        roles_present = sorted({roles.get(v, "covariate")
                                for v in result.variables})
        _add_legend_cluster(
            dot,
            roles_present  = roles_present,
            has_bidirected = bool(result.bidirected_edges),
            has_flagged    = bool(flagged & set(result.directed_edges)),
            has_undirected = bool(result.undirected_edges),
            rank_with      = [v for v in result.variables if v in _SOURCE_NODES],
        )

    if save_path:
        _render_dot(dot, save_path)

    return dot


def plot_edge_list(
    edges: Iterable[tuple],
    title: Optional[str] = None,
    variables: Optional[list] = None,
    node_roles: Optional[dict] = None,
    flagged_edges: Optional[Iterable[Tuple[str, str]]] = None,
    proxy_edges: Optional[Iterable[Tuple[str, str]]] = None,
    show_coefficients: bool = True,
    save_path: Optional[str] = None,
    show_legend: bool = True,
    preset: str = DEFAULT_PRESET,
    flagged_label: str = "Flagged (for review)",
):
    """
    Render a KNOWN graph — one specified by hand rather than discovered.

    ``plot_discovery_result`` takes a ``DiscoveryResult``, which only an
    algorithm produces.  A ground-truth SCM is just an edge list, so this
    wraps that list in a ``DiscoveryResult`` and hands it to the same
    ``_build_dot`` the discovered graphs go through.  Node shapes, fills,
    legend, and the PDF+PNG output are therefore identical by construction:
    there is one renderer, not two.

    Parameters
    ----------
    edges : iterable of ``(src, dst)`` or ``(src, dst, label)``
        Directed edges.  The optional third element is the edge label, taken
        verbatim — pass the SCM's own coefficient strings.
    variables : list, optional
        Node order.  Defaults to order of first appearance in ``edges``.
    node_roles : dict, optional
        ``{node: role}``; unlisted nodes fall back to "covariate".  Use the
        "latent" role for unobserved nodes — it draws a dashed diamond, and
        edges leaving it are drawn thin/dashed to match.
    flagged_edges : iterable, optional
        Drawn red dashed, exactly as in the discovery figures.
    proxy_edges : iterable, optional
        Drawn in the proxy colour with a heavier stroke.

    Returns the ``graphviz.Digraph``.
    """
    _check_graphviz()

    edges = [tuple(e) for e in edges]
    pairs  = [(e[0], e[1]) for e in edges]
    labels = {(e[0], e[1]): e[2] for e in edges if len(e) > 2}

    if variables is None:
        variables = []
        for src, dst in pairs:
            for v in (src, dst):
                if v not in variables:
                    variables.append(v)

    flagged = set(flagged_edges or ())
    proxy   = set(proxy_edges or ())
    roles   = node_roles or {v: "covariate" for v in variables}

    result = DiscoveryResult(
        algorithm      = title or "ground truth",
        variables      = list(variables),
        directed_edges = pairs,
    )

    dot = _build_dot(
        result            = result,
        title             = title or "",
        flagged           = flagged,
        roles             = roles,
        show_coefficients = show_coefficients,
        coef_threshold    = 0.0,
        preset            = preset,
        show_title        = bool(title),
        proxy             = proxy,
        edge_labels       = labels,
    )

    if show_legend:
        roles_present = sorted({roles.get(v, "covariate") for v in variables})
        _add_legend_cluster(
            dot,
            roles_present   = roles_present,
            has_bidirected  = False,
            has_flagged     = bool(flagged & set(pairs)),
            has_undirected  = False,
            has_proxy       = bool(proxy & set(pairs)),
            has_latent_edge = any(roles.get(s) == "latent" for s, _ in pairs),
            flagged_label   = flagged_label,
            rank_with       = [v for v in variables if v in _SOURCE_NODES],
        )

    if save_path:
        _render_dot(dot, save_path)

    return dot


def plot_grid(
    results: dict,
    flagged_edges: Optional[Iterable[Tuple[str, str]]] = None,
    node_roles: Optional[dict] = None,
    title: str = "Discovered DAGs",
    save_path: Optional[str] = None,
    layout: str = "fixed",
    pos: Optional[dict] = None,
    figsize_per_panel: Tuple[float, float] = (8.0, 8.0),   #    figsize_per_panel: Tuple[float, float] = (7.5, 5.0),
    panel_titles: Optional[dict] = None,     # ← new
    coef_decimals: int = 4,                  # match the paper's table precision
):
    """
    Render all algorithm results in a 3-column grid.

    Each panel is rendered by Graphviz independently (so labels are always
    on their edges), then the PNG outputs are tiled into a single matplotlib
    figure for the overview grid.

    Panel height
    ------------
    Only ``figsize_per_panel[0]`` (the panel *width*) is honored directly.
    Each grid row is as tall as the aspect ratio Graphviz actually produced.
    These DAGs lay out ``rankdir=LR`` — wide and short — so a fixed square
    panel box parked every drawing in the middle of a box roughly three times
    too tall, which is what made the grid look vertically stretched.
    ``figsize_per_panel[1]`` is used only as an upper bound on row height.

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
                coef_decimals     = coef_decimals,
                # `panel_titles` is drawn by matplotlib as each panel's axes
                # title.  Baking it into the PNG too printed it twice.
                show_title        = False,
                node_font_size    = 12,    # was 13
                edge_font_size    = 9,    # was 8
                node_width        = "1.5", # was 1.15
            )
            # No per-panel legend — a shared legend sits below the grid
            out_png = os.path.join(tmpdir, f"{alg_name}.png")
            _render_dot_to_png_only(dot, out_png)
            panel_pngs.append((alg_name, out_png))

        # ── Measure the panels ────────────────────────────────────────────────
        cols = 3
        rows = (n + cols - 1) // cols

        images = [plt.imread(p) if os.path.exists(p) else None
                  for _, p in panel_pngs]
        labels = [(panel_titles or {}).get(alg, alg) for alg, _ in panel_pngs]

        # height / width of each rendered drawing (fallback: a wide-ish DAG)
        aspects = [(im.shape[0] / im.shape[1]) if im is not None else 0.5
                   for im in images]

        panel_w   = figsize_per_panel[0]
        max_row_h = figsize_per_panel[1]          # hard cap, not a target

        def _row(seq, r):
            return seq[r * cols:(r + 1) * cols]

        # A row is exactly as tall as its tallest drawing.
        row_h = [min(panel_w * max(_row(aspects, r)), max_row_h)
                 for r in range(rows)]

        # Panel titles are drawn outside the axes, so their space lives in the
        # inter-row gap (and in the top margin for the first row).
        TITLE_LINE  = 0.22                        # in, per line at 12 pt
        row_title_h = [TITLE_LINE * max(l.count("\n") + 1 for l in _row(labels, r)) + 0.08
                       for r in range(rows)]

        # ── Shared legend ─────────────────────────────────────────────────────
        roles_present = sorted({roles.get(v, "covariate")
                                 for _, res in valid
                                 for v in res.variables})
        edge_handles = [
            Line2D([0],[0], color=COLOR_DIRECTED,   lw=1.6,
                   marker=">", markersize=8, label="Directed (i → j)"),
            Line2D([0],[0], color=COLOR_UNDIRECTED, lw=1.5, linestyle="solid",
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

        # ── Compose panels into a matplotlib figure ───────────────────────────
        # Every band below is sized in inches and only then converted to the
        # figure fractions subplots_adjust/legend want.  Sizing in fractions
        # (top=0.90, bottom=0.05, ...) is what produced the huge empty bands:
        # a fraction of a very tall figure is a very tall margin.
        SUPTITLE_H  = 0.55                                    # in
        legend_ncol = min(len(all_handles), 2)
        legend_rows = math.ceil(len(all_handles) / legend_ncol)
        legend_h    = 0.20 * legend_rows + 0.30               # in

        row_gap  = max(row_title_h[1:]) if rows > 1 else 0.0
        top_band = SUPTITLE_H + row_title_h[0]
        fig_w    = panel_w * cols
        fig_h    = top_band + sum(row_h) + row_gap * (rows - 1) + legend_h

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs  = fig.add_gridspec(
            rows, cols,
            height_ratios = row_h,
            top    = 1.0 - top_band / fig_h,
            bottom = legend_h / fig_h,
            left   = 0.01,
            right  = 0.99,
            # hspace is a fraction of the *average* axes height.
            hspace = (row_gap / (sum(row_h) / rows)) if rows > 1 else 0.0,
            wspace = 0.03,
        )

        axes = [fig.add_subplot(gs[i // cols, i % cols])
                for i in range(rows * cols)]

        for ax, img, panel_label in zip(axes, images, labels):
            if img is not None:
                ax.imshow(img, interpolation="lanczos")
            ax.set_title(panel_label, fontsize=12, fontweight="bold", pad=4)
            ax.axis("off")

        for ax in axes[n:]:
            ax.set_axis_off()

        fig.suptitle(title, fontsize=14, fontweight="bold",
                     y=1.0 - (SUPTITLE_H / 2) / fig_h, va="center")
        fig.legend(
            handles=all_handles,
            loc="lower center",
            ncol=legend_ncol,
            frameon=True, fancybox=True, framealpha=0.95,
            edgecolor="#cccccc", fontsize=9,
            bbox_to_anchor=(0.5, 0.10 / fig_h),
        )

        if save_path:
            save_figure_dual_format(fig, save_path)

        return fig

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
