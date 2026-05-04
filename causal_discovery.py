"""
causal_discovery.py
===================
Unified wrappers around the six causal discovery algorithms used in the paper:

    Constraint-based : PC, FCI
    Score-based      : GES
    Permutation-based: GRaSP
    Functional causal: ICA-LiNGAM, DirectLiNGAM

Each wrapper returns a `DiscoveryResult` with:
    - directed_edges        : list[tuple[str,str]]    i -> j
    - undirected_edges      : list[tuple[str,str]]    i -- j
    - bidirected_edges      : list[tuple[str,str]]    i <-> j  (latent confounder)
    - coef_matrix           : np.ndarray | None       (LiNGAM only) B[i,j] = effect of j -> i
    - causal_order          : list[str] | None        (LiNGAM only)
    - raw                   : the underlying object returned by causal-learn

Edge-encoding convention used by causal-learn for PC / FCI / GES / GRaSP:
    G.graph[i, j] = -1, G.graph[j, i] = +1   ==>  i -> j   (directed)
    G.graph[i, j] = -1, G.graph[j, i] = -1   ==>  i -- j   (undirected)
    G.graph[i, j] = +1, G.graph[j, i] = +1   ==>  i <-> j  (bidirected, FCI only)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Apply NumPy 2.x compatibility shim BEFORE any causal-learn module is
# imported. causal-learn's GES/GRaSP score functions call np.mat (removed
# in NumPy 2.0) at module-import time, so the shim has to be loaded first.
# The shim is idempotent and a no-op on NumPy 1.x. See numpy_compat.py.
import numpy_compat  # noqa: F401

# causal-learn is imported lazily inside each algorithm wrapper so that
# visualization / SHD / DiscoveryResult work even if causal-learn is not yet
# installed. The import errors will only fire when an algorithm is actually
# called.
def _import_causal_learn():
    try:
        from causallearn.search.ConstraintBased.PC import pc as cl_pc
        from causallearn.search.ConstraintBased.FCI import fci as cl_fci
        from causallearn.search.ScoreBased.GES import ges as cl_ges
        from causallearn.search.PermutationBased.GRaSP import grasp as cl_grasp
        from causallearn.search.FCMBased import lingam as cl_lingam

        # Apply the post-import scoring patch (fixes the NumPy 2.x
        # ``float(yX @ XX_inv @ yX.T)`` failure inside GES/GRaSP). The
        # patch is idempotent and only affects causal-learn's scoring
        # module namespace, not the builtin float() anywhere else.
        numpy_compat.patch_causal_learn_scoring(verbose=False)

        return cl_pc, cl_fci, cl_ges, cl_grasp, cl_lingam
    except ImportError as e:
        raise ImportError(
            "causal-learn is not installed. Run `pip install causal-learn` "
            "(or `pip install -r requirements.txt`) before running the "
            "discovery algorithms."
        ) from e


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------
@dataclass
class DiscoveryResult:
    algorithm: str
    variables: List[str]
    directed_edges: List[Tuple[str, str]] = field(default_factory=list)
    undirected_edges: List[Tuple[str, str]] = field(default_factory=list)
    bidirected_edges: List[Tuple[str, str]] = field(default_factory=list)
    coef_matrix: Optional[np.ndarray] = None
    causal_order: Optional[List[str]] = None
    raw: object = None

    def has_directed_edge(self, src: str, dst: str) -> bool:
        return (src, dst) in self.directed_edges

    def get_coefficient(self, src: str, dst: str) -> Optional[float]:
        """Return LiNGAM coefficient for src -> dst, or None if unavailable."""
        if self.coef_matrix is None:
            return None
        i_src = self.variables.index(src)
        j_dst = self.variables.index(dst)
        # B[i, j] = effect of j -> i in causal-learn's lingam
        return float(self.coef_matrix[j_dst, i_src])

    def summary(self) -> str:
        lines = [f"=== {self.algorithm} ==="]
        if self.directed_edges:
            lines.append("Directed edges (i -> j):")
            for s, d in self.directed_edges:
                coef = self.get_coefficient(s, d)
                tag = f"  beta={coef:+.4f}" if coef is not None else ""
                lines.append(f"  {s:>10s} -> {d:<10s}{tag}")
        if self.undirected_edges:
            lines.append("Undirected edges (i -- j):")
            for s, d in self.undirected_edges:
                lines.append(f"  {s:>10s} -- {d:<10s}")
        if self.bidirected_edges:
            lines.append("Bidirected edges (i <-> j) -- LATENT CONFOUNDER suspected:")
            for s, d in self.bidirected_edges:
                lines.append(f"  {s:>10s} <-> {d:<10s}")
        if self.causal_order is not None:
            lines.append(f"Causal order: {' -> '.join(self.causal_order)}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Helper: extract edges from a causal-learn GeneralGraph
# ------------------------------------------------------------------
def _extract_edges_from_graph(graph_matrix: np.ndarray, variables: List[str]):
    """Convert a causal-learn graph matrix to edge lists.

    Causal-learn's encoding convention:

        graph[i, j] stores the ENDPOINT MARK at node i for the edge
        between i and j (NOT at node j).

        Endpoint values:  -1 = TAIL,  +1 = ARROW

    Therefore:
        i -> j   :  graph[i,j] = -1 (tail at i),    graph[j,i] = +1 (arrowhead at j)
        j -> i   :  graph[i,j] = +1 (arrowhead at i), graph[j,i] = -1 (tail at j)
        i -- j   :  graph[i,j] = -1, graph[j,i] = -1   (tails at both ends)
        i <-> j  :  graph[i,j] = +1, graph[j,i] = +1   (arrowheads at both ends)

    Verified by:
      - The PyWhy community blog (Ken Koon Wong, 2023) which transposes
        causal-learn matrices before visualization specifically because
        their orientation is inverted from the standard adjacency matrix.
      - causal-learn's own TestGraphVisualization.py, which constructs an
        Edge with (TAIL, ARROW) endpoints in (source, target) order.
    """
    directed, undirected, bidirected = [], [], []
    n = len(variables)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = graph_matrix[i, j], graph_matrix[j, i]
            if a == 0 and b == 0:
                continue
            key = frozenset((i, j))
            if key in seen:
                continue
            seen.add(key)
            # i -> j : tail at i (-1), arrowhead at j (+1)
            if a == -1 and b == 1:
                directed.append((variables[i], variables[j]))
            # j -> i : arrowhead at i (+1), tail at j (-1)
            elif a == 1 and b == -1:
                directed.append((variables[j], variables[i]))
            # bidirected (FCI): arrowheads at both ends (+1, +1)
            elif a == 1 and b == 1:
                bidirected.append((variables[i], variables[j]))
            # undirected: tails at both ends (-1, -1)
            elif a == -1 and b == -1:
                undirected.append((variables[i], variables[j]))
            else:
                # Partially oriented (FCI circle marks etc.) -- treat as undirected
                undirected.append((variables[i], variables[j]))
    return directed, undirected, bidirected


# ------------------------------------------------------------------
# Algorithm wrappers
# ------------------------------------------------------------------
def run_pc(data: pd.DataFrame, alpha: float = 0.05) -> DiscoveryResult:
    cl_pc, *_ = _import_causal_learn()
    variables = list(data.columns)
    # `show_progress` is supported in recent versions; fall back if not.
    try:
        cg = cl_pc(data.values, alpha=alpha, indep_test="fisherz", show_progress=False)
    except TypeError:
        cg = cl_pc(data.values, alpha, "fisherz")
    directed, undirected, bidirected = _extract_edges_from_graph(cg.G.graph, variables)
    return DiscoveryResult(
        algorithm="PC",
        variables=variables,
        directed_edges=directed,
        undirected_edges=undirected,
        bidirected_edges=bidirected,
        raw=cg,
    )


def run_fci(data: pd.DataFrame, alpha: float = 0.05) -> DiscoveryResult:
    _, cl_fci, *_ = _import_causal_learn()
    variables = list(data.columns)
    try:
        g, _edges = cl_fci(data.values, alpha=alpha, indep_test="fisherz", verbose=False)
    except TypeError:
        # older signatures take positional only
        g, _edges = cl_fci(data.values, "fisherz", alpha)
    directed, undirected, bidirected = _extract_edges_from_graph(g.graph, variables)
    return DiscoveryResult(
        algorithm="FCI",
        variables=variables,
        directed_edges=directed,
        undirected_edges=undirected,
        bidirected_edges=bidirected,
        raw=g,
    )


def run_ges(data: pd.DataFrame) -> DiscoveryResult:
    _, _, cl_ges, *_ = _import_causal_learn()
    variables = list(data.columns)
    res = cl_ges(data.values, score_func="local_score_BIC")
    G = res["G"]
    directed, undirected, bidirected = _extract_edges_from_graph(G.graph, variables)
    return DiscoveryResult(
        algorithm="GES",
        variables=variables,
        directed_edges=directed,
        undirected_edges=undirected,
        bidirected_edges=bidirected,
        raw=res,
    )


def run_grasp(data: pd.DataFrame) -> DiscoveryResult:
    _, _, _, cl_grasp, _ = _import_causal_learn()
    variables = list(data.columns)
    G = cl_grasp(data.values, score_func="local_score_BIC")
    graph_matrix = G.graph if hasattr(G, "graph") else G
    directed, undirected, bidirected = _extract_edges_from_graph(graph_matrix, variables)
    return DiscoveryResult(
        algorithm="GRaSP",
        variables=variables,
        directed_edges=directed,
        undirected_edges=undirected,
        bidirected_edges=bidirected,
        raw=G,
    )


def _lingam_to_result(model, variables: List[str], algorithm: str) -> DiscoveryResult:
    """Convert a fitted LiNGAM model to a DiscoveryResult."""
    B = model.adjacency_matrix_           # B[i, j] = coefficient on j -> i
    order_idx = list(model.causal_order_)  # indices in causal order
    causal_order = [variables[i] for i in order_idx]

    # Threshold tiny coefficients to avoid numerical noise edges
    threshold = 1e-3
    directed = []
    n = B.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(B[i, j]) > threshold:
                # j -> i with coefficient B[i, j]
                directed.append((variables[j], variables[i]))

    return DiscoveryResult(
        algorithm=algorithm,
        variables=variables,
        directed_edges=directed,
        coef_matrix=B,
        causal_order=causal_order,
        raw=model,
    )


def _build_prior_knowledge_matrix(
    variables: List[str],
    exogenous: Optional[List[str]] = None,
    sinks: Optional[List[str]] = None,
    forbidden_edges: Optional[List[Tuple[str, str]]] = None,
) -> Optional[np.ndarray]:
    """Construct a causal-learn prior_knowledge matrix.

    Convention from LiNGAM official docs (verified against the
    visualization helper at lingam.readthedocs.io/tutorial/pk_direct):

        pk[i, j]  = 1  -> edge j -> i is REQUIRED
        pk[i, j]  = 0  -> edge j -> i is FORBIDDEN
        pk[i, j]  = -1 -> unknown

    Note carefully: the FIRST index is the DESTINATION (i.e., the row tells
    you which variable receives the arrow), and the SECOND index is the
    SOURCE (the column tells you which variable sends the arrow). This is
    the same convention as the LiNGAM ``adjacency_matrix_`` output, so it
    is internally consistent within the package -- but it is the OPPOSITE
    of the more common 'pk[from, to]' convention used by some other
    libraries. Get this wrong and the algorithm will treat your sources
    as sinks and vice versa.

    Therefore:
        * To make ``v`` EXOGENOUS (no incoming edges):
          forbid every edge ``other -> v``. That means ``pk[v, other] = 0``
          for every ``other != v``. In matrix terms: zero ROW v (off-diag).

        * To make ``v`` a SINK (no outgoing edges):
          forbid every edge ``v -> other``. That means ``pk[other, v] = 0``
          for every ``other != v``. In matrix terms: zero COLUMN v (off-diag).

    Parameters
    ----------
    variables : list of column names in the data, in order.
    exogenous : variables that should have NO incoming edges
        (i.e., they are root causes -- Race, Sex, Age, etc.).
    sinks : variables that should have NO outgoing edges
        (i.e., they are pure outcomes -- Recidivism, Loan, Score, etc.).
    forbidden_edges : explicit (src, dst) pairs to forbid.

    Returns ``None`` when no prior knowledge is provided, so the LiNGAM
    call falls back to the unrestricted estimation.
    """
    if not exogenous and not sinks and not forbidden_edges:
        return None

    n = len(variables)
    pk = -np.ones((n, n), dtype=int)  # -1 = unknown (matches docs convention)
    # Note: we do NOT zero the diagonal. The docs example keeps diagonals
    # at -1, and LiNGAM ignores self-loops anyway.
    idx = {v: i for i, v in enumerate(variables)}

    for src in (exogenous or []):
        if src not in idx:
            continue
        # Nothing has a path INTO an exogenous variable.
        # "Edge other -> src forbidden" means pk[src, other] = 0.
        # So zero ROW src (off-diagonal; diagonal stays -1).
        v = idx[src]
        for j in range(n):
            if j != v:
                pk[v, j] = 0

    for snk in (sinks or []):
        if snk not in idx:
            continue
        # Nothing flows OUT OF a sink variable.
        # "Edge snk -> other forbidden" means pk[other, snk] = 0.
        # So zero COLUMN snk (off-diagonal; diagonal stays -1).
        v = idx[snk]
        for i in range(n):
            if i != v:
                pk[i, v] = 0

    for src, dst in (forbidden_edges or []):
        # "Edge src -> dst forbidden" means pk[dst, src] = 0.
        if src in idx and dst in idx:
            pk[idx[dst], idx[src]] = 0

    return pk


def run_direct_lingam(
    data: pd.DataFrame,
    exogenous: Optional[List[str]] = None,
    sinks: Optional[List[str]] = None,
    forbidden_edges: Optional[List[Tuple[str, str]]] = None,
) -> DiscoveryResult:
    """Run DirectLiNGAM, optionally with prior knowledge.

    Pass ``exogenous`` / ``sinks`` / ``forbidden_edges`` for real-world
    datasets where LiNGAM's continuous-noise assumption is violated.
    Without prior knowledge, the algorithm can produce orderings that
    contradict basic domain knowledge (e.g., ChargeDegree -> Race).
    """
    *_, cl_lingam = _import_causal_learn()
    variables = list(data.columns)
    pk = _build_prior_knowledge_matrix(variables, exogenous, sinks, forbidden_edges)
    if pk is not None:
        model = cl_lingam.DirectLiNGAM(prior_knowledge=pk)
    else:
        model = cl_lingam.DirectLiNGAM()
    model.fit(data.values)
    return _lingam_to_result(model, variables, algorithm="DirectLiNGAM")


def run_ica_lingam(
    data: pd.DataFrame,
    random_state: int = 42,
) -> DiscoveryResult:
    """Run ICA-LiNGAM. ICA-LiNGAM does not accept prior knowledge in
    causal-learn (only DirectLiNGAM does), so this wrapper takes no
    knowledge arguments. If you need prior knowledge on COMPAS-like
    data, use ``run_direct_lingam`` instead.
    """
    *_, cl_lingam = _import_causal_learn()
    variables = list(data.columns)
    model = cl_lingam.ICALiNGAM(random_state=random_state)
    model.fit(data.values)
    return _lingam_to_result(model, variables, algorithm="ICA-LiNGAM")


# ------------------------------------------------------------------
# Convenience: run them all
# ------------------------------------------------------------------
ALGORITHMS = {
    "PC": run_pc,
    "FCI": run_fci,
    "GES": run_ges,
    "GRaSP": run_grasp,
    "ICA-LiNGAM": run_ica_lingam,
    "DirectLiNGAM": run_direct_lingam,
}


def run_all(
    data: pd.DataFrame,
    skip: Optional[List[str]] = None,
    verbose_errors: bool = True,
    direct_lingam_exogenous: Optional[List[str]] = None,
    direct_lingam_sinks: Optional[List[str]] = None,
    direct_lingam_forbidden_edges: Optional[List[Tuple[str, str]]] = None,
) -> dict:
    """Run every algorithm and return {name: DiscoveryResult}.

    When ``verbose_errors=True`` (default), failures print the full
    traceback so you can see exactly which causal-learn function and
    which line of its code raised the error.

    The ``direct_lingam_*`` parameters are forwarded only to
    DirectLiNGAM and let you fix exogenous variables / sinks for
    datasets where LiNGAM's noise assumption fails (typical for
    real-world data with discrete variables, e.g. COMPAS). Pass
    ``direct_lingam_exogenous=['Race', 'Sex', 'Age']`` to anchor
    immutable demographics as roots of the causal order. ICA-LiNGAM
    cannot use prior knowledge in causal-learn and runs unconstrained.
    """
    import traceback
    skip = set(skip or [])
    results = {}

    use_pk = bool(
        direct_lingam_exogenous
        or direct_lingam_sinks
        or direct_lingam_forbidden_edges
    )

    for name, fn in ALGORITHMS.items():
        if name in skip:
            continue
        try:
            if name == "DirectLiNGAM" and use_pk:
                results[name] = run_direct_lingam(
                    data,
                    exogenous=direct_lingam_exogenous,
                    sinks=direct_lingam_sinks,
                    forbidden_edges=direct_lingam_forbidden_edges,
                )
            else:
                results[name] = fn(data)
        except Exception as exc:
            if verbose_errors:
                print()
                print("=" * 70)
                print(f"[FAIL] {name} raised {type(exc).__name__}: {exc}")
                print("-" * 70)
                print("Full traceback (copy this when reporting the issue):")
                traceback.print_exc()
                print("=" * 70)
            else:
                print(f"[WARN] {name} failed: {type(exc).__name__}: {exc}")
            results[name] = None
    return results


# ------------------------------------------------------------------
# Structural Hamming Distance
# ------------------------------------------------------------------
def structural_hamming_distance(
    result: DiscoveryResult, ground_truth_edges: List[Tuple[str, str]]
) -> int:
    """SHD = #(edges added) + #(edges deleted) + #(edges reversed).

    Both directed and undirected predicted edges are counted; an undirected
    prediction is treated as a wrong orientation (counts as a reversal-style
    error if the true edge has the opposite direction; otherwise as an extra
    edge).
    """
    truth = set(ground_truth_edges)
    pred_directed = set(result.directed_edges)
    pred_undirected = {frozenset(e) for e in result.undirected_edges}

    shd = 0
    # Edges in prediction not in truth (extras + reversals)
    for s, d in pred_directed:
        if (s, d) in truth:
            continue
        if (d, s) in truth:
            shd += 1   # reversed orientation
        else:
            shd += 1   # spurious edge
    # Undirected predictions count as one error each
    for fs in pred_undirected:
        a, b = tuple(fs)
        if (a, b) not in truth and (b, a) not in truth:
            shd += 1
        else:
            shd += 1   # truth is directed, prediction is undirected
    # Edges in truth missing from prediction
    pred_undirected_pairs = {tuple(fs) for fs in pred_undirected}
    pred_undirected_pairs |= {tuple(reversed(p)) for p in pred_undirected_pairs}
    for s, d in truth:
        if (s, d) in pred_directed:
            continue
        if (s, d) in pred_undirected_pairs:
            continue   # already counted above
        shd += 1
    return shd


if __name__ == "__main__":
    from synthetic_data import generate_paired_datasets

    A, B = generate_paired_datasets(n=1000, beta_biased=-0.15, seed=42)
    print("Running all algorithms on Dataset A (biased)...\n")
    results = run_all(A)
    for name, res in results.items():
        if res is None:
            continue
        print(res.summary())
        if res.has_directed_edge("Race", "Loan"):
            print(">>> Race -> Loan DETECTED <<<")
        print()
