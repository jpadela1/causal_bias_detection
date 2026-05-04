"""
visualization.py
================
Plot the causal graphs produced by the discovery algorithms.

- Directed edges  (i -> j)  : solid arrows
- Undirected      (i -- j)  : grey dotted lines (no arrowhead)
- Bidirected      (i <-> j) : red dashed double-arrows (latent confounder)
- A specific "flagged" edge (e.g. Race -> Loan, Race -> Score) is highlighted
  in red to draw the auditor's eye to the bias path.

Layout philosophy
-----------------
matplotlib's auto-layout ignores node circles when it computes axis limits.
We therefore reserve explicit space along all four edges of each axes for
node padding, and reserve a fixed strip at the bottom of every figure for
the legend. This means nodes never sit on the figure edge, edges never
disappear under the legend, and the legend never sits on top of the graph.

Why this file does NOT use draw_networkx_edges
----------------------------------------------
NetworkX's ``draw_networkx_edges`` has a long history of arrows-not-rendering
issues that depend on the matplotlib + networkx version pair, the value of
the ``node_size`` keyword (which must match the actual node draw size), and
the chosen connectionstyle. We sidestep all of that by drawing edges directly
with matplotlib's ``FancyArrowPatch`` -- the object networkx uses internally
anyway -- with ``shrinkB`` computed from the actual node radius.

Output formats
--------------
We always save both PNG (raster, 300 dpi) and PDF (vector, infinitely
zoomable) for every plot.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

from causal_discovery import DiscoveryResult


# --- Sizes (kept as constants so node draw and arrow shrink stay in sync) ---
NODE_SIZE_LARGE = 1700   # for plot_discovery_result (single big plot)
NODE_SIZE_SMALL = 900    # for plot_grid (six panels)


def _node_radius_pt(node_size: float) -> float:
    """matplotlib scatter ``s`` is area in pt^2 -> radius in pt."""
    return math.sqrt(node_size / math.pi)
# --- Color scheme (matches the paper's figures) ----------------------------
NODE_ROLE_COLORS = {
    "protected": "#8ab4d4",  # blue — Race, Sex/Gender 5B8FF9
    "proxy": "#f0a868",  # orange — ZIP, Priors F6BD16
    "mediator": "#90c88c",  # green — Income, CreditSc, Juv_Fel, Juv_Misd 5AD8A6
    "outcome": "#e8a0a0",  # pink — LoanApprv, COMPAS_Score, Recidivism F08BB4
    "covariate": "#c8c8c8",  # grey — Age, Charge, Education 9DA0A4
}

# Human-readable role labels for the legend.
ROLE_LABELS = {
    "protected": "Protected attribute",
    "proxy":     "Proxy variable",
    "mediator":  "Mediator",
    "outcome":   "Outcome",
    "covariate": "Covariate",
}

DEFAULT_ROLES_LOAN = {
    "Race": "protected",
    "Gender": "protected",
    "ZIP": "proxy",
    "Education": "covariate",
    "Income": "mediator",
    "CreditSc": "mediator",
    "Loan": "outcome",
}

DEFAULT_ROLES_COMPAS = {
    "Race": "protected",
    "Sex": "protected",
    "Age": "covariate",
    "JuvFelony": "mediator",
    "JuvMisd": "mediator",
    "Priors": "proxy",
    "ChargeDegree": "covariate",
    "Score": "outcome",
    "Recidivism": "outcome",
}


# --- Layout ---------------------------------------------------------------
def _resolve_node_overlaps(
    pos: dict,
    min_distance: float,
    max_iterations: int = 60,
) -> dict:
    """Push apart any pair of nodes that are closer than ``min_distance``.

    Iterative pairwise repulsion: each iteration finds all overlapping
    pairs and shoves them apart along the line connecting them. Converges
    quickly because the perturbations get smaller as nodes spread out.
    Returns the same dict object (also mutates in place).
    """
    keys = list(pos.keys())
    if len(keys) < 2:
        return pos
    for _ in range(max_iterations):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                xa, ya = pos[a]
                xb, yb = pos[b]
                dx, dy = xb - xa, yb - ya
                dist = math.hypot(dx, dy)
                if dist >= min_distance:
                    continue
                # Push apart along the connecting line; if perfectly
                # coincident, jitter in an arbitrary direction.
                if dist < 1e-9:
                    dx, dy, dist = 1.0, 0.0, 1.0
                shove = (min_distance - dist) / 2.0 + 1e-3
                ux, uy = dx / dist, dy / dist
                pos[a] = (xa - ux * shove, ya - uy * shove)
                pos[b] = (xb + ux * shove, yb + uy * shove)
                moved = True
        if not moved:
            break
    return pos


def _min_node_distance_for_layout(node_size: float) -> float:
    """Approximate node-circle radius in *layout* coordinates.

    Layouts return positions roughly in [-1, 1]^2. A node drawn with
    matplotlib scatter at size ``s`` (pt^2) has visual radius
    sqrt(s/pi) points. For a typical 9x6 inch figure at ~80 dpi inside
    a 2x2 layout box, one layout-unit ~ 200 points. We use a conservative
    factor that empirically prevents visual overlap across our figure
    sizes.
    """
    radius_pt = _node_radius_pt(node_size)
    # Empirically tuned: this is the minimum center-to-center distance in
    # layout units (~ [-1, 1] range) that prevents the rendered circles
    # from touching across figsizes (9x8) and (5.5x5.0). Adjust if you
    # change figure sizes substantially.
    return radius_pt / 110.0


def _layout(
    g: nx.Graph,
    layout: str = "auto",
    seed: int = 7,
    node_size: float = NODE_SIZE_LARGE,
    full_graph: Optional[nx.Graph] = None,
):
    """Pick a layout, then resolve node-on-node overlaps.

    'auto' uses a layered top-down layout for graphs whose directed-edge
    structure dominates (>= 60% of edges directed AND graph is acyclic),
    and kamada-kawai over the full skeleton otherwise.

    Parameters
    ----------
    g : DiGraph of just the directed edges.
    full_graph : Graph including ALL edges (directed + undirected +
        bidirected, ignoring orientation). Used by kamada-kawai so the
        layout reflects the full skeleton, not just the oriented part.
        Falls back to ``g`` if not provided.
    """
    n = g.number_of_nodes()
    skeleton = full_graph if full_graph is not None else g

    if layout == "auto":
        # Count edges in the skeleton vs in the directed-only g.
        n_directed = g.number_of_edges()
        n_total = skeleton.number_of_edges()
        directed_fraction = (n_directed / n_total) if n_total > 0 else 0.0

        is_dag = (
            isinstance(g, nx.DiGraph)
            and n_directed > 0
            and nx.is_directed_acyclic_graph(g)
        )
        if is_dag and directed_fraction >= 0.6:
            layout = "layered"
        elif n >= 7:
            layout = "kamada"
        else:
            layout = "spring"

    if layout == "spring":
        pos = nx.spring_layout(
            skeleton, seed=seed,
            k=2.5 / math.sqrt(max(n, 1)), iterations=200,
        )
    elif layout == "kamada":
        # kamada_kawai needs a connected graph. If skeleton is disconnected,
        # we add a tiny imaginary star to glue components together so the
        # layout converges, then ignore the imaginary node afterwards.
        if nx.is_connected(skeleton.to_undirected()
                           if skeleton.is_directed() else skeleton):
            pos = nx.kamada_kawai_layout(skeleton)
        else:
            pos = _kamada_with_glue(skeleton)
    elif layout == "layered":
        pos = _layered_layout(g)
    elif layout == "circular":
        pos = nx.circular_layout(skeleton)
    elif layout == "shell":
        pos = nx.shell_layout(skeleton)
    else:
        raise ValueError(f"Unknown layout {layout}")

    pos = {k: (float(v[0]), float(v[1])) for k, v in pos.items()}
    min_dist = _min_node_distance_for_layout(node_size)
    _resolve_node_overlaps(pos, min_distance=min_dist)
    return pos


def _kamada_with_glue(g: nx.Graph) -> dict:
    """Run kamada-kawai on a disconnected graph by temporarily adding a
    weakly-weighted glue node connecting all components."""
    g_und = g.to_undirected() if g.is_directed() else g.copy()
    glue = "__glue__"
    g_und.add_node(glue)
    for v in list(g_und.nodes()):
        if v != glue:
            g_und.add_edge(glue, v, weight=0.01)
    pos = nx.kamada_kawai_layout(g_und)
    pos.pop(glue, None)
    return pos


def _layered_layout(g: nx.DiGraph) -> dict:
    """Place nodes in horizontal layers by topological rank.

    Sources at the top (y high), sinks at the bottom. Nodes within the
    same rank are spaced evenly along x. Reads naturally for causal DAGs
    where you want to see information flow.
    """
    # nx.topological_generations groups nodes whose ancestors are in
    # earlier generations. Result: a list of lists (each sublist = one rank).
    try:
        generations = list(nx.topological_generations(g))
    except nx.NetworkXUnfeasible:
        # Has a cycle; fall back to kamada.
        return nx.kamada_kawai_layout(g)

    if not generations:
        return {}

    n_layers = len(generations)
    pos = {}
    for layer_idx, nodes in enumerate(generations):
        # y from +1 (top) down to -1 (bottom)
        if n_layers == 1:
            y = 0.0
        else:
            y = 1.0 - 2.0 * layer_idx / (n_layers - 1)
        # x evenly spaced in [-1, 1]
        k = len(nodes)
        for i, v in enumerate(sorted(nodes)):
            if k == 1:
                x = 0.0
            else:
                x = -1.0 + 2.0 * i / (k - 1)
            pos[v] = (x, y)
    return pos


def _set_axes_with_padding(ax, pos: dict, pad_frac: float = 0.30) -> None:
    """Compute axis limits with generous padding so node circles don't
    clip the axes box. Default 30% padding on every side."""
    if not pos:
        return
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(xs) - min(xs) or 1.0
    span_y = max(ys) - min(ys) or 1.0
    pad_x = span_x * pad_frac
    pad_y = span_y * pad_frac
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    # NOTE: we deliberately do NOT call set_aspect('equal') -- it can
    # override our explicit xlim/ylim when matplotlib decides to "fulfill"
    # the aspect ratio, leaving big white margins or clipping nodes.


# --- Drawing primitives ----------------------------------------------------
def _draw_directed_arrow(
    ax,
    pos_src,
    pos_dst,
    color: str = "#333333",
    linewidth: float = 1.6,
    arrowsize: float = 18,
    style: str = "solid",
    rad: float = 0.08,
    node_size: float = NODE_SIZE_LARGE,
    extra_margin_pt: float = 4.0,
    zorder: int = 2,
):
    """Draw a single directed arrow from src to dst with proper shrinkback.

    ``shrinkB`` is the number of points the arrow tip is pulled back from
    the target endpoint. We add ``extra_margin_pt`` so the tip sits cleanly
    outside the node circle rather than touching it.
    """
    radius_pt = _node_radius_pt(node_size)
    arrow = FancyArrowPatch(
        posA=pos_src,
        posB=pos_dst,
        arrowstyle="-|>",
        mutation_scale=arrowsize,
        color=color,
        linewidth=linewidth,
        linestyle=style,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=radius_pt + extra_margin_pt,
        shrinkB=radius_pt + extra_margin_pt,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _draw_undirected_line(
    ax,
    pos_src,
    pos_dst,
    color: str = "#888888",
    linewidth: float = 1.2,
    style: str = "dotted",
    node_size: float = NODE_SIZE_LARGE,
    extra_margin_pt: float = 4.0,
    zorder: int = 1,
):
    """Plain line, no arrowhead, with the same shrinkback as the arrows."""
    radius_pt = _node_radius_pt(node_size)
    line = FancyArrowPatch(
        posA=pos_src,
        posB=pos_dst,
        arrowstyle="-",
        mutation_scale=1,
        color=color,
        linewidth=linewidth,
        linestyle=style,
        connectionstyle="arc3,rad=0",
        shrinkA=radius_pt + extra_margin_pt,
        shrinkB=radius_pt + extra_margin_pt,
        zorder=zorder,
    )
    ax.add_patch(line)
    return line


# --- Legend builders ------------------------------------------------------
def _build_edge_legend_handles():
    """Three line styles: directed, undirected, bidirected/flagged."""
    return [
        Line2D([0], [0], color="#333333", lw=1.6, marker=">",
               markersize=8, label="Directed (i \u2192 j)"),
        Line2D([0], [0], color="#888888", lw=1.0, linestyle=":",
               label="Undirected (i \u2014 j)"),
        Line2D([0], [0], color="#D62728", lw=1.6, linestyle="--",
               marker=">", markersize=8,
               label="Bidirected / flagged"),
    ]


def _build_node_legend_handles(roles_present):
    """Color swatches for each role that actually appears in the plot."""
    return [
        Patch(facecolor=NODE_ROLE_COLORS[role], edgecolor="black",
              label=ROLE_LABELS[role])
        for role in roles_present
    ]


# --- I/O ------------------------------------------------------------------
def save_figure_dual_format(fig, save_path: str, dpi: int = 300) -> None:
    """Save the figure as both PNG (raster) and PDF (vector).

    ``save_path`` may include any extension (or none). We strip the
    extension and write both 'foo.png' and 'foo.pdf'.
    """
    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    png_path = base + ".png"
    pdf_path = base + ".pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  saved: {png_path}")
    print(f"  saved: {pdf_path}")


# Backward-compat alias
_save_both_formats = save_figure_dual_format


def _print_edge_summary(result: DiscoveryResult) -> None:
    n_dir = len(result.directed_edges)
    n_und = len(result.undirected_edges)
    n_bi = len(result.bidirected_edges)
    total = n_dir + n_und + n_bi
    if total == 0:
        print(f"  [{result.algorithm}] 0 edges discovered")
        return
    pct = 100.0 * n_dir / total
    print(
        f"  [{result.algorithm}] {n_dir} directed, {n_und} undirected, "
        f"{n_bi} bidirected  ({pct:.0f}% oriented)"
    )


# --- Public plot functions -------------------------------------------------
def plot_discovery_result(
    result: DiscoveryResult,
    title: Optional[str] = None,
    flagged_edges: Optional[Iterable[Tuple[str, str]]] = None,
    node_roles: Optional[dict] = None,
    show_coefficients: bool = True,
    coef_threshold: float = 0.05,
    layout: str = "auto",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (11, 8),
    show_legend: bool = True,
):
    """Render a discovery result as a labeled DAG with edge & node legend.

    Saves both PNG and PDF when ``save_path`` is provided. The extension on
    ``save_path`` is ignored.
    """
    flagged = set(flagged_edges or [])
    roles = node_roles or {v: "covariate" for v in result.variables}

    g = nx.DiGraph()
    for v in result.variables:
        g.add_node(v)
    for s, d in result.directed_edges:
        g.add_edge(s, d)

    # Skeleton graph (undirected, all edge types) — used by kamada_kawai
    # so the layout uses the FULL connectivity, not just oriented edges.
    skeleton = nx.Graph()
    skeleton.add_nodes_from(result.variables)
    for s, d in result.directed_edges:
        skeleton.add_edge(s, d)
    for s, d in result.undirected_edges:
        skeleton.add_edge(s, d)
    for s, d in result.bidirected_edges:
        skeleton.add_edge(s, d)

    pos = _layout(g, layout=layout, node_size=NODE_SIZE_LARGE,
                  full_graph=skeleton)
    _print_edge_summary(result)

    fig, ax = plt.subplots(figsize=figsize)

    # --- Edges first ------------------------------------------------------
    for s, d in result.directed_edges:
        is_flagged = (s, d) in flagged
        _draw_directed_arrow(
            ax, pos[s], pos[d],
            color="#D62728" if is_flagged else "#333333",
            linewidth=2.6 if is_flagged else 1.6,
            arrowsize=26 if is_flagged else 20,
            style="dashed" if is_flagged else "solid",
            node_size=NODE_SIZE_LARGE,
            zorder=3 if is_flagged else 2,
        )
    for s, d in result.undirected_edges:
        _draw_undirected_line(
            ax, pos[s], pos[d],
            color="#888888", linewidth=1.3, style="dotted",
            node_size=NODE_SIZE_LARGE,
        )
    for s, d in result.bidirected_edges:
        _draw_directed_arrow(
            ax, pos[s], pos[d], color="#D62728", linewidth=1.8,
            arrowsize=16, style="dashed", rad=0.15,
            node_size=NODE_SIZE_LARGE,
        )
        _draw_directed_arrow(
            ax, pos[d], pos[s], color="#D62728", linewidth=1.8,
            arrowsize=16, style="dashed", rad=0.15,
            node_size=NODE_SIZE_LARGE,
        )

    # --- Nodes ------------------------------------------------------------
    roles_present = []
    seen_roles = set()
    for v in result.variables:
        role = roles.get(v, "covariate")
        if role not in seen_roles:
            roles_present.append(role)
            seen_roles.add(role)
        ax.scatter(
            pos[v][0], pos[v][1],
            s=NODE_SIZE_LARGE,
            c=NODE_ROLE_COLORS[role],
            edgecolors="black",
            linewidths=1.5,
            zorder=4,
        )
        ax.annotate(
            v, xy=pos[v], ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=5,
        )

    # --- LiNGAM coefficient annotations -----------------------------------
    if show_coefficients and result.coef_matrix is not None:
        for s, d in result.directed_edges:
            coef = result.get_coefficient(s, d)
            if coef is None or abs(coef) < coef_threshold:
                continue
            mid_x = (pos[s][0] + pos[d][0]) / 2
            mid_y = (pos[s][1] + pos[d][1]) / 2
            ax.text(
                mid_x, mid_y, f"{coef:+.3f}",
                fontsize=7, ha="center", va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          boxstyle="round,pad=0.18"),
            )

    # --- Frame ------------------------------------------------------------
    ax.set_title(title or result.algorithm, fontsize=13, fontweight="bold",
                 pad=14)
    ax.set_axis_off()
    _set_axes_with_padding(ax, pos, pad_frac=0.18)

    # --- Legend (reserves bottom strip of figure) -------------------------
    if show_legend:
        edge_handles = _build_edge_legend_handles()
        node_handles = _build_node_legend_handles(roles_present)

        # Reserve 18% of the figure for the legend; place it via the figure
        # (not the axes) so axis padding logic above is unaffected.
        fig.subplots_adjust(bottom=0.18)
        all_handles = edge_handles + node_handles
        fig.legend(
            handles=all_handles,
            loc="lower center",
            ncol=min(len(all_handles), 4),
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            edgecolor="#cccccc",
            fontsize=9,
            bbox_to_anchor=(0.5, 0.02),
        )
    else:
        fig.tight_layout()

    if save_path:
        save_figure_dual_format(fig, save_path)
    return fig, ax


def plot_grid(
    results: dict,
    flagged_edges: Optional[Iterable[Tuple[str, str]]] = None,
    node_roles: Optional[dict] = None,
    title: str = "Discovered DAGs",
    save_path: Optional[str] = None,
    layout: str = "auto",
    figsize_per_panel: Tuple[float, float] = (5.5, 5.0),
):
    """Plot all algorithm results in one figure (rows of 3) with a legend."""
    valid = [(k, v) for k, v in results.items() if v is not None]
    n = len(valid)
    if n == 0:
        print("No valid results to plot.")
        return None

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    flagged = set(flagged_edges or [])
    roles = node_roles or {}

    seen_roles_global = set()
    for ax, (name, res) in zip(axes, valid):
        g = nx.DiGraph()
        for v in res.variables:
            g.add_node(v)
        for s, d in res.directed_edges:
            g.add_edge(s, d)

        skeleton = nx.Graph()
        skeleton.add_nodes_from(res.variables)
        for s, d in res.directed_edges:
            skeleton.add_edge(s, d)
        for s, d in res.undirected_edges:
            skeleton.add_edge(s, d)
        for s, d in res.bidirected_edges:
            skeleton.add_edge(s, d)

        pos = _layout(g, layout=layout, node_size=NODE_SIZE_SMALL,
                      full_graph=skeleton)

        # Edges first
        for s, d in res.directed_edges:
            is_flagged = (s, d) in flagged
            _draw_directed_arrow(
                ax, pos[s], pos[d],
                color="#D62728" if is_flagged else "#333333",
                linewidth=2.2 if is_flagged else 1.1,
                arrowsize=18 if is_flagged else 13,
                style="dashed" if is_flagged else "solid",
                node_size=NODE_SIZE_SMALL,
                extra_margin_pt=2.0,
                zorder=3 if is_flagged else 2,
            )
        for s, d in res.undirected_edges:
            _draw_undirected_line(
                ax, pos[s], pos[d],
                color="#888", linewidth=0.9, style="dotted",
                node_size=NODE_SIZE_SMALL, extra_margin_pt=2.0,
            )
        for s, d in res.bidirected_edges:
            _draw_directed_arrow(
                ax, pos[s], pos[d], color="#D62728",
                linewidth=1.3, arrowsize=11, style="dashed", rad=0.15,
                node_size=NODE_SIZE_SMALL, extra_margin_pt=2.0,
            )
            _draw_directed_arrow(
                ax, pos[d], pos[s], color="#D62728",
                linewidth=1.3, arrowsize=11, style="dashed", rad=0.15,
                node_size=NODE_SIZE_SMALL, extra_margin_pt=2.0,
            )

        # Nodes
        for v in res.variables:
            role = roles.get(v, "covariate")
            seen_roles_global.add(role)
            ax.scatter(
                pos[v][0], pos[v][1], s=NODE_SIZE_SMALL,
                c=NODE_ROLE_COLORS[role], edgecolors="black",
                linewidths=1.0, zorder=4,
            )
            ax.annotate(
                v, xy=pos[v], ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=5,
            )

        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        ax.set_axis_off()
        _set_axes_with_padding(ax, pos, pad_frac=0.18)

    for ax in axes[n:]:
        ax.set_axis_off()

    # --- Combined edge + node legend at the bottom of the figure ----------
    # Order roles in the legend to match the original NODE_ROLE_COLORS order.
    roles_present = [r for r in NODE_ROLE_COLORS if r in seen_roles_global]
    edge_handles = _build_edge_legend_handles()
    node_handles = _build_node_legend_handles(roles_present)
    all_handles = edge_handles + node_handles

    # Reserve a fixed strip at the bottom for the legend, then place it.
    legend_strip = 0.10 + 0.012 * (len(all_handles) // 4)  # grows if needed
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.93, bottom=max(legend_strip, 0.08),
                        left=0.03, right=0.97, hspace=0.18, wspace=0.10)
    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=min(len(all_handles), 4),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    if save_path:
        save_figure_dual_format(fig, save_path)
    return fig
