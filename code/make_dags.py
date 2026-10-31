#!/usr/bin/env python3
"""
make_dags.py -- regenerate the causal-graph figures (1, 4, 5) with Graphviz.

Everything is driven by CSV files, so a figure is reproducible from data you
already have rather than from remembered layout coordinates. No LaTeX, no
matplotlib, no spring layout.

WHY GRAPHVIZ HERE AND NOT FOR FIG. 2
------------------------------------
Graphviz lays out graphs. Figure 2 is a bar chart -- it has axes, a numeric
scale and bars, none of which Graphviz has any concept of. Figure 2 stays in
matplotlib (see make_fig2.py). Figures 1, 4 and 5 are graphs, and Graphviz's
`dot` engine does exactly the layered left-to-right arrangement these DAGs
want.

FONT EMBEDDING
--------------
Graphviz's PDF output (cairo) embeds fonts as TrueType, NOT Type 3. Verified
with `pdffonts`. So this route is safe for IEEE PDF eXpress, unlike the
matplotlib defaults that produced the submitted figures.

--------------------------------------------------------------------------
FILE FORMATS
--------------------------------------------------------------------------
nodes CSV   name,label,role,layer
            role  : protected | proxy | mediator | covariate | outcome | latent
            layer : integer; nodes sharing a layer are ranked together, which
                    is what makes the layout reproducible instead of organic

edges CSV   from,to,kind,weight,label,flag
            kind  : directed | bidirected | undirected
            weight: float, may be blank
            label : text to print on the edge; blank means no label
            flag  : 1 draws it red and dashed (the paths the text discusses)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
  # Figure 1 -- ground-truth SCM, biased dataset only
  python make_dags.py build --nodes data/nodes_synth.csv \\
                            --edges data/edges_fig1.csv \\
                            --out figures/ground_truth_dag

  # Figure 5 -- DirectLiNGAM on COMPAS
  python make_dags.py build --nodes data/nodes_compas.csv \\
                            --edges data/edges_dlingam.csv \\
                            --out figures/compas_direct_lingam --legend

  # Figure 4 -- one PDF per panel, placed side by side in the LaTeX float
  python make_dags.py build --nodes data/nodes_compas.csv \\
      --edges data/edges_fci.csv   --out figures/panel_fci   --title "FCI"
  python make_dags.py build --nodes data/nodes_compas.csv \\
      --edges data/edges_grasp.csv --out figures/panel_grasp --title "GRaSP"
  python make_dags.py build --nodes data/nodes_compas.csv \\
      --edges data/edges_dlingam.csv --out figures/panel_dlingam \\
      --title "DirectLiNGAM"

  # Turn a saved result matrix into an edges CSV
  python make_dags.py convert --format lingam --matrix results/dlingam_B.csv \\
      --names Race,Sex,Age,JuvFel,JuvMisd,Charge,Priors,Score,Recid \\
      --label Race:Score --label Race:Priors --label Priors:Score \\
      --flag  Race:Score --flag  Race:Priors \\
      --out data/edges_dlingam.csv

In LaTeX, nothing but graphicx is needed:

    \\begin{figure}[t]\\centering
      \\includegraphics[width=\\columnwidth]{figures/ground_truth_dag.pdf}
      \\caption{...}
    \\end{figure}

    \\begin{figure*}[t]\\centering
      \\includegraphics[width=0.32\\textwidth]{figures/panel_fci.pdf}\\hfill
      \\includegraphics[width=0.32\\textwidth]{figures/panel_grasp.pdf}\\hfill
      \\includegraphics[width=0.32\\textwidth]{figures/panel_dlingam.pdf}
      \\caption{...}
    \\end{figure*}
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pydot

# Node colours: the same semantics used throughout the paper.
ROLE_FILL = {
    "protected": "#A9CCE3",
    "proxy":     "#F5B041",
    "mediator":  "#A9DFBF",
    "covariate": "#D0D3D4",
    "outcome":   "#F5B7B1",
    "latent":    "#FFFFFF",
}
ROLE_ORDER = ["protected", "proxy", "mediator", "covariate", "outcome", "latent"]
ROLE_CAPTION = {
    "protected": "Protected attribute",
    "proxy":     "Proxy discriminator",
    "mediator":  "Mediator",
    "covariate": "Covariate",
    "outcome":   "Outcome",
    "latent":    "Latent (unobserved)",
}

FLAG_COLOR = "#C0392B"
EDGE_COLOR = "#2B2B2B"
MUTED_COLOR = "#7F8C8D"


# ----------------------------------------------------------------- reading ---
def read_nodes(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("name", "").strip():
                continue
            role = (r.get("role") or "covariate").strip().lower()
            if role not in ROLE_FILL:
                sys.exit(f"{path}: unknown role {role!r} for node {r['name']!r}. "
                         f"Use one of {sorted(ROLE_FILL)}")
            rows.append({
                "name": r["name"].strip(),
                "label": (r.get("label") or r["name"]).strip(),
                "role": role,
                "layer": int(r["layer"]) if (r.get("layer") or "").strip() else None,
            })
    if not rows:
        sys.exit(f"{path}: no nodes")
    return rows


def read_edges(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("from", "").strip():
                continue
            w = (r.get("weight") or "").strip()
            rows.append({
                "from": r["from"].strip(),
                "to": r["to"].strip(),
                "kind": (r.get("kind") or "directed").strip().lower(),
                "weight": float(w) if w else None,
                "label": (r.get("label") or "").strip(),
                "flag": str(r.get("flag") or "0").strip() in ("1", "true", "True", "yes"),
            })
    return rows


# ------------------------------------------------------------------ build ----
def build(args):
    nodes = read_nodes(args.nodes)
    edges = read_edges(args.edges)
    known = {n["name"] for n in nodes}
    for e in edges:
        for endpoint in (e["from"], e["to"]):
            if endpoint not in known:
                sys.exit(f"{args.edges}: edge endpoint {endpoint!r} is not in "
                         f"{args.nodes}")

    g = pydot.Dot(
        graph_type="digraph",
        rankdir=args.rankdir,
        splines="spline",
        overlap="false",
        nodesep=str(args.nodesep),
        ranksep=str(args.ranksep),
        fontname=args.font,
        fontsize=str(args.title_size),
        margin="0.02",
        bgcolor="transparent",
    )
    if args.title:
        g.set("label", args.title)
        g.set("labelloc", "t")
        g.set("labeljust", "c")

    g.set_node_defaults(
        shape="circle", style="filled", fixedsize="true",
        width=str(args.node_size), height=str(args.node_size),
        fontname=args.font, fontsize=str(args.node_font),
        penwidth="0.8", color="#1B1B1B",
    )
    g.set_edge_defaults(
        fontname=args.font, fontsize=str(args.edge_font),
        color=EDGE_COLOR, penwidth="0.8", arrowsize="0.6",
    )

    for n in nodes:
        attrs = dict(label=n["label"].replace("\\n", "\n"),
                     fillcolor=ROLE_FILL[n["role"]])
        if n["role"] == "latent":
            attrs.update(style="filled,dashed", color=MUTED_COLOR,
                         fontcolor=MUTED_COLOR)
        g.add_node(pydot.Node(n["name"], **attrs))

    # rank=same groups make the layout deterministic and layered
    layers = defaultdict(list)
    for n in nodes:
        if n["layer"] is not None:
            layers[n["layer"]].append(n["name"])
    for _, members in sorted(layers.items()):
        if len(members) > 1:
            sub = pydot.Subgraph(rank="same")
            for m in members:
                sub.add_node(pydot.Node(m))
            g.add_subgraph(sub)

    latent_names = {n["name"] for n in nodes if n["role"] == "latent"}

    for e in edges:
        attrs = {}
        if e["from"] in latent_names or e["to"] in latent_names:
            attrs.update(color=MUTED_COLOR, fontcolor=MUTED_COLOR,
                         style="dashed", penwidth="0.6")
        if e["kind"] == "bidirected":
            attrs["dir"] = "both"
        elif e["kind"] == "undirected":
            attrs["dir"] = "none"
        elif e["kind"] != "directed":
            sys.exit(f"{args.edges}: unknown kind {e['kind']!r}")

        label = e["label"]
        if not label and e["weight"] is not None and args.label_all:
            label = f"{e['weight']:+.{args.decimals}f}"
        if label:
            attrs["label"] = label

        if e["flag"]:
            attrs.update(color=FLAG_COLOR, style="dashed", penwidth="1.4",
                         fontcolor=FLAG_COLOR)
        elif (args.mute_unlabelled and not label
              and e["from"] not in latent_names
              and e["to"] not in latent_names):
            attrs.update(color=MUTED_COLOR, penwidth="0.6")

        g.add_edge(pydot.Edge(e["from"], e["to"], **attrs))

    if args.legend:
        used = [r for r in ROLE_ORDER if any(n["role"] == r for n in nodes)]
        leg = pydot.Subgraph("cluster_legend", rank="same",
                             label="", color="transparent", style="invis",
                             fontsize=str(args.edge_font), fontname=args.font)
        prev = None
        for role in used:
            nid = f"_leg_{role}"
            leg.add_node(pydot.Node(
                nid, label=ROLE_CAPTION[role], shape="box", style="filled",
                fillcolor=ROLE_FILL[role], fixedsize="false",
                width="0", height="0", margin="0.04,0.02",
                fontsize=str(args.edge_font), fontname=args.font,
                color="#1B1B1B", penwidth="0.6"))
            if prev:
                leg.add_edge(pydot.Edge(prev, nid, style="invis"))
            prev = nid
        g.add_subgraph(leg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out.with_suffix(".dot")
    dot_path.write_text(g.to_string(), encoding="utf-8")

    for fmt, extra in (("pdf", []), ("png", ["-Gdpi=300"])):
        target = out.with_suffix("." + fmt)
        cmd = [args.engine, f"-T{fmt}", *extra, str(dot_path), "-o", str(target)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            sys.exit(f"'{args.engine}' not found. Install Graphviz: "
                     f"apt-get install graphviz / brew install graphviz")
        except subprocess.CalledProcessError as exc:
            sys.exit(f"graphviz failed:\n{exc.stderr}")
        print(f"wrote {target}")
    print(f"wrote {dot_path}  (the DOT source; keep it, it is the figure's "
          f"reproducible form)")
    ver = subprocess.run([args.engine, "-V"], capture_output=True, text=True)
    print((ver.stderr or ver.stdout).strip())
    print("verify with:  pdffonts " + str(out.with_suffix('.pdf'))
          + "   -> no row should say 'Type 3'")


# ---------------------------------------------------------------- convert ----
def read_matrix(path):
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    return [[float(c) for c in r] for r in rows]


def convert(args):
    names = [s.strip() for s in args.names.split(",")]
    m = read_matrix(args.matrix)
    if len(m) != len(names):
        sys.exit(f"matrix is {len(m)}x{len(m[0])} but {len(names)} names given")

    def pairs(items):
        out = set()
        for s in items:
            a, _, b = s.partition(":")
            out.add((a.strip(), b.strip()))
        return out

    labelled, flagged = pairs(args.label), pairs(args.flag)
    edges = []

    if args.format == "lingam":
        # B[i][j] is the coefficient on the edge j -> i
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                w = m[i][j]
                if i != j and abs(w) >= args.threshold:
                    edges.append((name_j, name_i, "directed", w))
    else:  # causal-learn GeneralGraph.graph
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = m[i][j], m[j][i]
                if a == 0 and b == 0:
                    continue
                if a == -1 and b == 1:
                    edges.append((names[i], names[j], "directed", None))
                elif b == -1 and a == 1:
                    edges.append((names[j], names[i], "directed", None))
                elif a == 1 and b == 1:
                    edges.append((names[i], names[j], "bidirected", None))
                else:
                    edges.append((names[i], names[j], "undirected", None))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_lab = n_flag = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["from", "to", "kind", "weight", "label", "flag"])
        for src, dst, kind, weight in edges:
            is_flag = (src, dst) in flagged or (dst, src) in flagged
            want_lab = (src, dst) in labelled or (dst, src) in labelled
            label = (f"{weight:+.{args.decimals}f}"
                     if want_lab and weight is not None else "")
            n_lab += bool(label)
            n_flag += is_flag
            w.writerow([src, dst, kind,
                        f"{weight:.6f}" if weight is not None else "",
                        label, int(is_flag)])
    print(f"wrote {out}: {len(edges)} edges, {n_flag} flagged, {n_lab} labelled")
    if labelled and not n_lab:
        print("  note: --label had no effect; those edges are not in the matrix",
              file=sys.stderr)


# ------------------------------------------------------------------- cli -----
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="render a DAG from nodes+edges CSVs")
    b.add_argument("--nodes", required=True)
    b.add_argument("--edges", required=True)
    b.add_argument("--out", required=True, help="output path without extension")
    b.add_argument("--title", default="")
    b.add_argument("--engine", default="dot",
                   choices=["dot", "neato", "fdp", "sfdp", "circo", "twopi"])
    b.add_argument("--rankdir", default="LR", choices=["LR", "TB", "RL", "BT"])
    b.add_argument("--font", default="Helvetica")
    b.add_argument("--node-size", type=float, default=0.62,
                   help="node diameter in inches")
    b.add_argument("--node-font", type=float, default=8.0)
    b.add_argument("--edge-font", type=float, default=7.0)
    b.add_argument("--title-size", type=float, default=10.0)
    b.add_argument("--nodesep", type=float, default=0.28)
    b.add_argument("--ranksep", type=float, default=0.55)
    b.add_argument("--decimals", type=int, default=4)
    b.add_argument("--label-all", action="store_true",
                   help="print every weight. Off by default: eighteen labels "
                        "is what made the submitted Fig. 5 unreadable")
    b.add_argument("--mute-unlabelled", action="store_true", default=True)
    b.add_argument("--no-mute-unlabelled", dest="mute_unlabelled",
                   action="store_false")
    b.add_argument("--legend", action="store_true")
    b.set_defaults(func=build)

    c = sub.add_parser("convert", help="result matrix -> edges CSV")
    c.add_argument("--format", required=True, choices=["causal-learn", "lingam"])
    c.add_argument("--matrix", required=True)
    c.add_argument("--names", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--label", action="append", default=[], metavar="FROM:TO")
    c.add_argument("--flag", action="append", default=[], metavar="FROM:TO")
    c.add_argument("--threshold", type=float, default=0.01)
    c.add_argument("--decimals", type=int, default=4)
    c.set_defaults(func=convert)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
