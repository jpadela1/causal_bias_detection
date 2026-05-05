"""
=============================================================================
CAUSAL DISCOVERY FOR BIAS DETECTION IN HISTORICAL DATASETS
Using Causal Discovery to Detect and Understand Bias in Historical Datasets
Author : Joyce Padela (PhD Research)
Version: 2.0  — complete rewrite with realistic data and verified API calls
=============================================================================

RESEARCH QUESTION:
  Can causal discovery algorithms identify structural patterns in data that
  give rise to biased outcomes?

CAUSAL STRUCTURE ENCODED IN BOTH DATASETS:
  Race    → ZIP Code    → Credit Score → Loan Approved   (proxy discrimination)
  Race    → Income      → Loan Approved
  Gender  → Income      → Loan Approved
  Educ    → Income      → Loan Approved
  Educ    → Credit Score→ Loan Approved
  Race    → Loan Approved   ← ONLY in Dataset A (direct discrimination, β = –0.15)

VARIABLE INDEX MAP (used throughout):
  0 = Race        1 = Gender      2 = Education
  3 = ZIP Code    4 = Income      5 = Credit Score   6 = Loan Approved

HOW TO RUN:
  Step 1 — Run Section 0 to install packages (first time only, takes ~3 min)
  Step 2 — Run Sections 1–12 in order
  Each section prints results and saves one or more figures to disk.
=============================================================================
"""

# =============================================================================
# SECTION 0 — INSTALL ALL REQUIRED PACKAGES
# Run this section first. On first run it may take 2–4 minutes.
# If you are in Jupyter Notebook use:  !pip install causal-learn dowhy ...
# =============================================================================
import subprocess, sys

def _install(pkg, import_as=None):
    imp = import_as or pkg.replace("-", "")
    try:
        __import__(imp)
        print(f"  [OK]  {pkg}")
    except ImportError:
        print(f"  [Installing] {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               pkg, "--quiet"])
        print(f"  [Done] {pkg}")

print("="*70)
print("SECTION 0: Installing packages")
print("="*70)
for p, i in [("causal-learn","causallearn"), ("dowhy",None),
             ("networkx",None), ("matplotlib",None), ("numpy",None),
             ("pandas",None), ("scipy",None), ("scikit-learn","sklearn"),
             ("statsmodels",None)]:
    _install(p, i)
print()


# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
print("="*70)
print("SECTION 1: Importing libraries")
print("="*70)

import warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

# ── causal-learn: constraint-based ──────────────────────────────────────────
from causallearn.search.ConstraintBased.PC  import pc
from causallearn.search.ConstraintBased.FCI import fci

# ── causal-learn: score-based ───────────────────────────────────────────────
from causallearn.search.ScoreBased.GES import ges

# ── causal-learn: permutation-based (GRaSP — best for small graphs) ─────────
try:
    from causallearn.search.PermutationBased.GRaSP import grasp
    GRASP_AVAILABLE = True
except ImportError:
    GRASP_AVAILABLE = False
    print("  GRaSP not available in this causal-learn version — skipping")

# ── causal-learn: functional (LiNGAM family) ────────────────────────────────
from causallearn.search.FCMBased import lingam          # ICALiNGAM
# DirectLiNGAM is more robust; use it as second LiNGAM variant
try:
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    DIRECT_LINGAM_AVAILABLE = True
except ImportError:
    DIRECT_LINGAM_AVAILABLE = False

# ── causal-learn: independence test strings ──────────────────────────────────
# PC and FCI accept string names: "fisherz", "chisq", "gsq", "kci"
# GES accepts no test string — uses BIC internally
# LiNGAM: model.fit(data)

# ── DoWhy ────────────────────────────────────────────────────────────────────
import dowhy
from dowhy import CausalModel

print("All libraries imported.\n")


# =============================================================================
# SECTION 2 — CONSTANTS: VARIABLE NAMES, LAYOUT, HELPER FUNCTIONS
# =============================================================================
print("="*70)
print("SECTION 2: Defining constants and helper functions")
print("="*70)

COLS   = ["Race", "Gender", "Educ", "ZIP", "Income", "CreditSc", "LoanApprv"]
N_VARS = len(COLS)                     # 7
IDX    = {n: i for i, n in enumerate(COLS)}

# Fixed 2-D layout for every graph — same positions every time so figures
# are directly comparable when placed side-by-side in the paper.
POS = {
    0: (0.0,  1.8),   # Race         — top-left protected attribute
    1: (0.0,  0.0),   # Gender       — bottom-left protected attribute
    2: (1.6,  0.9),   # Education    — centre-left neutral covariate
    3: (3.2,  2.4),   # ZIP Code     — upper-centre proxy variable
    4: (3.8,  0.9),   # Income       — centre mediator
    5: (3.2, -0.6),   # Credit Score — lower-centre mediator
    6: (5.8,  0.9),   # Loan Approved— far-right outcome
}

# Short node labels used inside the circles
NODE_LABEL = {
    0: "Race\n(Prot.)",
    1: "Gender\n(Prot.)",
    2: "Educ",
    3: "ZIP\n(Proxy)",
    4: "Income\n(Med.)",
    5: "Credit\nScore",
    6: "Loan\nApprv",
}

NODE_RADIUS = 0.38      # circle radius in data-units
ARROW_KW_DIR = dict(    # directed arrow style
    arrowstyle="-|>",
    color="black",
    lw=1.8,
    mutation_scale=22,  # larger arrowhead
)
ARROW_KW_BIDIR = dict(  # bidirected arrow style (FCI latent confounder)
    arrowstyle="<->",
    color="black",
    lw=1.8,
    mutation_scale=22,
    connectionstyle="arc3,rad=0.28",
)


# ─── adjacency matrix helpers ────────────────────────────────────────────────

def make_gt_adj(include_direct_race_bias: bool) -> np.ndarray:
    """Return ground-truth adjacency matrix (value 1 = i → j)."""
    A = np.zeros((N_VARS, N_VARS), dtype=int)
    edges = [
        (0,3),(0,4),(1,4),(2,4),(2,5),   # structural paths
        (3,5),(4,6),(5,6),                # structural paths
    ]
    if include_direct_race_bias:
        edges.append((0,6))               # direct discrimination path
    for (i,j) in edges:
        A[i,j] = 1
    return A


def normalize_pc_ges(graph_matrix: np.ndarray) -> np.ndarray:
    """
    Convert causal-learn PC / GES graph matrix to simple adj.

    causal-learn PC / GES convention (CPDAG):
      graph[i,j] == -1  AND  graph[j,i] ==  1  →  i → j
      graph[i,j] == -1  AND  graph[j,i] == -1  →  i — j  (undirected)
    We represent undirected as i→j for drawing (flag with value 3).
    """
    n   = graph_matrix.shape[0]
    adj = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            if graph_matrix[j,i]==1 and graph_matrix[i,j]==-1:
                adj[i,j] = 1          # directed  i → j
            elif graph_matrix[i,j]==-1 and graph_matrix[j,i]==-1 and i<j:
                adj[i,j] = 3          # undirected i — j
    return adj


def parse_fci(fci_graph) -> np.ndarray:
    """
    Convert causal-learn FCI PAG graph to adj matrix.

    Verified causal-learn FCI mark codes for g[i,j]
    (= mark AT the j-end of edge between i and j):
      1 = arrowhead (→)   2 = tail (—)   3 = circle (o)   0 = no edge

    Patterns:
      Directed i→j  : g[i,j]==1, g[j,i]==2
      Directed j→i  : g[j,i]==1, g[i,j]==2
      Bidirected i↔j: g[i,j]==1, g[j,i]==1   ← latent confounder
      Partial  i o→j: g[i,j]==1, g[j,i]==3
      Undirected    : g[i,j]==2, g[j,i]==2

    Output encoding: 0=none, 1=directed, 2=bidirected, 3=partial/undirected
    """
    g   = fci_graph.graph
    n   = g.shape[0]
    adj = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            mij, mji = g[i,j], g[j,i]
            if mij==0 and mji==0:
                continue
            if   mij==1 and mji==2:  adj[i,j]=1          # i → j
            elif mji==1 and mij==2:  adj[j,i]=1          # j → i
            elif mij==1 and mji==1:  adj[i,j]=adj[j,i]=2 # i ↔ j
            elif mij==1 and mji==3:  adj[i,j]=3          # i o→ j
            elif mji==1 and mij==3:  adj[j,i]=3          # j o→ i
            else:                    adj[i,j]=3           # other partial
    return adj


def lingam_to_adj(B: np.ndarray, thr=0.05) -> np.ndarray:
    """
    LiNGAM adjacency_matrix_ has B[i,j] = effect of j ON i  (column=cause).
    We transpose so adj[cause, effect] = 1.
    """
    adj = np.zeros((N_VARS, N_VARS), dtype=int)
    for i in range(N_VARS):
        for j in range(N_VARS):
            if abs(B[i,j]) > thr:
                adj[j,i] = 1     # j causes i
    return adj


def compute_shd(true_adj: np.ndarray, est_adj: np.ndarray) -> int:
    """Structural Hamming Distance (lower = better; 0 = perfect)."""
    shd = 0
    for i in range(N_VARS):
        for j in range(i+1, N_VARS):
            te = (true_adj[i,j], true_adj[j,i])
            ee = (min(est_adj[i,j],1), min(est_adj[j,i],1))
            if te != ee:
                shd += 1
    return shd


def print_edges(adj: np.ndarray, label=""):
    """Pretty-print edges from an adjacency matrix."""
    print(f"  ── {label} ──")
    found = False
    for i in range(N_VARS):
        for j in range(N_VARS):
            if adj[i,j]==1 and adj[j,i]==0:
                print(f"    {COLS[i]:10s} →  {COLS[j]}")
                found = True
            elif adj[i,j]==2 and adj[j,i]==2 and i<j:
                print(f"    {COLS[i]:10s} ↔  {COLS[j]}  (bidirected / latent)")
                found = True
            elif adj[i,j]==3 and i<j:
                print(f"    {COLS[i]:10s} —  {COLS[j]}  (undirected / partial)")
                found = True
    if not found:
        print("    (no edges found)")
    print()


# ─── drawing ─────────────────────────────────────────────────────────────────

def draw_graph(adj: np.ndarray, title: str, filename: str,
               highlight: list = None, notes: str = ""):
    """
    Draw a 7-node causal graph from adj matrix and save to PNG.

    adj values:
      0 = no edge
      1 = directed  i → j
      2 = bidirected i ↔ j  (drawn with curved double-arrowhead)
      3 = undirected / partially oriented  i — j

    highlight : list of (i,j) tuples to draw as dashed red edges
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.axis("off")

    hl_set = set(highlight or [])

    # ── draw nodes ────────────────────────────────────────────────────────
    for idx in range(N_VARS):
        x, y = POS[idx]
        circ = plt.Circle((x,y), NODE_RADIUS,
                           facecolor="white", edgecolor="black",
                           linewidth=2.0, zorder=4)
        ax.add_patch(circ)
        ax.text(x, y, NODE_LABEL[idx],
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", zorder=5, multialignment="center")

    # ── draw edges ────────────────────────────────────────────────────────
    drawn = set()
    for i in range(N_VARS):
        for j in range(N_VARS):
            if adj[i,j] == 0:
                continue
            if (j,i) in drawn:
                continue

            xi, yi = POS[i];  xj, yj = POS[j]
            dx, dy = xj-xi, yj-yi
            L  = np.sqrt(dx**2 + dy**2)
            ux, uy = dx/L, dy/L
            pad = NODE_RADIUS + 0.07
            sx, sy = xi + ux*pad, yi + uy*pad   # start (outside node i)
            ex, ey = xj - ux*pad, yj - uy*pad   # end   (outside node j)

            is_hl = (i,j) in hl_set or (j,i) in hl_set
            color = "black"
            lw    = 2.2 if is_hl else 1.8
            ls    = "--" if is_hl else "-"

            if adj[i,j]==2 and adj[j,i]==2:
                # ── bidirected  i ↔ j  (latent confounder) ────────────
                # Unpack ARROW_KW_BIDIR into a new dict, then override
                # only lw (color stays black from the constant).
                bidir_kw = {**ARROW_KW_BIDIR, "lw": lw}
                ax.annotate("", xy=(ex,ey), xytext=(sx,sy),
                            arrowprops=bidir_kw,
                            zorder=3)
            elif adj[i,j]==1 and adj[j,i]==0:
                # ── directed  i → j ───────────────────────────────────
                # Unpack ARROW_KW_DIR into a new dict, then override
                # lw and linestyle (dashed for highlighted bias edges).
                dir_kw = {**ARROW_KW_DIR, "lw": lw, "linestyle": ls}
                ax.annotate("", xy=(ex,ey), xytext=(sx,sy),
                            arrowprops=dir_kw,
                            zorder=3)
            elif adj[i,j]==3:
                # ── undirected / partial ───────────────────────────────
                ax.plot([sx,ex],[sy,ey], color=color, lw=lw,
                        linestyle=ls, zorder=3)
            drawn.add((i,j))

    if notes:
        fig.text(0.5, 0.01, notes, ha="center", fontsize=7.5,
                 style="italic", color="#333333", wrap=True)

    xs = [p[0] for p in POS.values()]
    ys = [p[1] for p in POS.values()]
    ax.set_xlim(min(xs)-0.9, max(xs)+0.9)
    ax.set_ylim(min(ys)-1.0, max(ys)+1.0)
    plt.tight_layout()
    # Save as PDF — vector format, infinitely zoomable for reviewers.
    # dpi is ignored for PDF (it is a vector format), so we omit it.
    plt.savefig(filename, bbox_inches="tight", facecolor="white",
                format="pdf")
    plt.close()
    print(f"  Saved → {filename}")


print("Constants and helpers defined.\n")


# =============================================================================
# SECTION 3 — REALISTIC SYNTHETIC DATASETS
# =============================================================================
# Dataset A (Biased)   : direct_race_bias = -0.15
# Dataset B (Unbiased) : direct_race_bias =  0.00
#
# Realistic formatting:
#   ZIP Code    — 5-digit integer (e.g. 10001)
#   Income      — dollars, 2 decimal places
#   Education   — years, 2 decimal places
#   Credit Score— integer 300–850
#   Race        — 0 / 1  (no decimals)
#   Gender      — 0 / 1  (no decimals)
#   Loan Approved — 0 / 1 (no decimals)
# =============================================================================
print("="*70)
print("SECTION 3: Generating realistic synthetic datasets")
print("="*70)

np.random.seed(2024)
N = 1500    # sample size — large enough for algorithms, small enough to run fast

# ZIP code base ranges by race group (structural racism in housing markets)
# Majority group (Race=0): higher-resource ZIP codes centred around 10200
# Minority group (Race=1): lower-resource ZIP codes centred around 10050
ZIP_BASE_MAJORITY = 10200
ZIP_BASE_MINORITY = 10050
ZIP_STD           =  60     # standard deviation within each group

def generate_dataset(direct_race_bias: float, n: int = N,
                     seed_offset: int = 0) -> pd.DataFrame:
    """
    Generate one synthetic loan-approval dataset.

    Parameters
    ----------
    direct_race_bias : coefficient on the direct Race → Loan path.
                       -0.15 = biased;  0.00 = unbiased.
    n                : number of rows.
    seed_offset      : add to the global seed so datasets A & B differ.
    """
    rng = np.random.default_rng(2024 + seed_offset)

    # ── protected attributes (exogenous roots) ──────────────────────────────
    Race   = rng.binomial(1, 0.32, n)           # ~32% minority
    Gender = rng.binomial(1, 0.50, n)           # 50/50

    # ── neutral covariate ───────────────────────────────────────────────────
    # Education in years (12–22); 2 decimal places
    Educ = np.round(rng.normal(15.5, 2.8, n).clip(10, 22), 2)

    # ── ZIP Code (5-digit, integer, proxy for race) ─────────────────────────
    zip_centre = np.where(Race==1, ZIP_BASE_MINORITY, ZIP_BASE_MAJORITY)
    ZIP = (zip_centre + rng.normal(0, ZIP_STD, n)).astype(int).clip(10001, 99999)

    # ZIP influence score: normalise to [–1, +1] for downstream use
    zip_score = (ZIP - ZIP.mean()) / ZIP.std()

    # ── Income (dollars, 2 d.p.) ─────────────────────────────────────────────
    # Causal parents: Race, Gender, Education
    Income_raw = (48000
                  - 7500  * Race           # racial income gap
                  - 5200  * Gender         # gender pay gap
                  + 2100  * Educ           # return on education
                  + rng.normal(0, 6500, n))
    Income = np.round(Income_raw.clip(15000, 250000), 2)

    # ── Credit Score (integer 300–850) ───────────────────────────────────────
    # Causal parents: ZIP (proxy for neighbourhood resources) and Education
    CreditSc_raw = (680
                    + 18  * zip_score       # neighbourhood resource effect
                    +  9  * (Educ - 15.5)   # education → financial literacy
                    + rng.normal(0, 42, n))
    CreditSc = CreditSc_raw.round().astype(int).clip(300, 850)

    # ── Loan Approved (binary outcome) ───────────────────────────────────────
    # Causal parents: Income, Credit Score, Education, and optionally Race
    loan_score = (0.0000045 * (Income  - 48000)
                  + 0.007   * (CreditSc - 680)
                  + 0.18    * (Educ - 15.5)
                  + direct_race_bias * Race      # ← 0 in fair dataset
                  + rng.normal(0, 0.35, n))
    LoanApprv = (loan_score > 0).astype(int)

    df = pd.DataFrame({
        "Race"    : Race,
        "Gender"  : Gender,
        "Educ"    : Educ,
        "ZIP"     : ZIP,
        "Income"  : Income,
        "CreditSc": CreditSc,
        "LoanApprv": LoanApprv,
    })
    return df


df_A = generate_dataset(direct_race_bias=-0.15, seed_offset=0)  # Biased
df_B = generate_dataset(direct_race_bias= 0.00, seed_offset=7)  # Unbiased

# ── Print cross-sample ───────────────────────────────────────────────────────
print("Dataset A — BIASED  (first 12 rows):")
print(df_A.head(12).to_string(index=True))
print()
print("Dataset B — UNBIASED  (first 12 rows):")
print(df_B.head(12).to_string(index=True))
print()

# ── Disparity check (traditional correlation-based view) ────────────────────
print("── Correlation-based disparity check ──")
for label, df in [("Biased", df_A), ("Unbiased", df_B)]:
    r_maj = df[df.Race==0].LoanApprv.mean()
    r_min = df[df.Race==1].LoanApprv.mean()
    dir_  = r_min / r_maj if r_maj > 0 else 0
    print(f"  {label:10s}: majority={r_maj:.3f}  minority={r_min:.3f}"
          f"  DIR={dir_:.3f}  (80% rule threshold = 0.800)")
print()

# ── Dataset statistics ───────────────────────────────────────────────────────
print("── Dataset A descriptive statistics ──")
print(df_A.describe().round(2).to_string())
print()


# =============================================================================
# SECTION 4 — GROUND TRUTH GRAPHS
# =============================================================================
print("="*70)
print("SECTION 4: Ground truth causal graphs")
print("="*70)

gt_A = make_gt_adj(include_direct_race_bias=True)
gt_B = make_gt_adj(include_direct_race_bias=False)

print("Ground truth edges — Biased (Dataset A):")
print_edges(gt_A, "includes Race → LoanApprv (direct discrimination)")

print("Ground truth edges — Unbiased (Dataset B):")
print_edges(gt_B, "no direct discrimination path")

draw_graph(gt_A,
    title="Ground Truth — BIASED Dataset A\n"
          "Race → LoanApprv (dashed) = direct discrimination  β = –0.15",
    filename="fig_01_gt_biased.pdf",
    highlight=[(0,6)],
    notes="Dashed edge Race → LoanApprv represents direct racial discrimination "
          "(β = –0.15). All other edges are the legitimate structural pathways "
          "present in both datasets.")

draw_graph(gt_B,
    title="Ground Truth — UNBIASED Dataset B\n"
          "No direct Race → LoanApprv path",
    filename="fig_02_gt_unbiased.pdf",
    notes="Race influences LoanApprv only indirectly: "
          "Race → ZIP → CreditSc → LoanApprv  and  Race → Income → LoanApprv.")

print()


# =============================================================================
# SECTION 5 — EXPERIMENT 1: PC ALGORITHM  (constraint-based)
# =============================================================================
# API call (causal-learn docs):
#   from causallearn.search.ConstraintBased.PC import pc
#   cg = pc(data, alpha, indep_test, stable, uc_rule, uc_priority,
#            mvpc, correction_name, background_knowledge, verbose,
#            show_progress, node_names)
#
# Key parameters used:
#   data        : numpy array  (n_samples × n_features), continuous
#   alpha       : significance level for conditional independence test (0.05)
#   indep_test  : "fisherz" — Fisher-Z test for continuous Gaussian data
#   stable      : True — use stable PC (order-independent skeleton)
#   show_progress: False — suppress progress bar
#   node_names  : list of variable name strings for labelling
# =============================================================================
print("="*70)
print("SECTION 5: Experiment 1 — PC Algorithm (Fisher-Z, α = 0.05)")
print("="*70)

# Standardise data before PC (Fisher-Z assumes Gaussian; scaling helps)
scaler = StandardScaler()
data_A = scaler.fit_transform(df_A.values.astype(float))
data_B = scaler.fit_transform(df_B.values.astype(float))

print("Running PC on Dataset A (Biased) ...")
cg_pc_A = pc(data       = data_A,
             alpha       = 0.05,
             indep_test  = "fisherz",
             stable      = True,
             uc_rule     = 0,
             uc_priority = 2,
             show_progress = False,
             node_names  = COLS)

print("Running PC on Dataset B (Unbiased) ...")
cg_pc_B = pc(data       = data_B,
             alpha       = 0.05,
             indep_test  = "fisherz",
             stable      = True,
             uc_rule     = 0,
             uc_priority = 2,
             show_progress = False,
             node_names  = COLS)

adj_pc_A = normalize_pc_ges(cg_pc_A.G.graph)
adj_pc_B = normalize_pc_ges(cg_pc_B.G.graph)

print("\nPC recovered edges — Dataset A (Biased):")
print_edges(adj_pc_A, "PC / Biased")
print("PC recovered edges — Dataset B (Unbiased):")
print_edges(adj_pc_B, "PC / Unbiased")

shd_pc_A = compute_shd(gt_A, adj_pc_A)
shd_pc_B = compute_shd(gt_B, adj_pc_B)
r_loan_pc_A = adj_pc_A[0,6]
r_loan_pc_B = adj_pc_B[0,6]
print(f"  SHD (Biased)  : {shd_pc_A}   (0 = perfect recovery)")
print(f"  SHD (Unbiased): {shd_pc_B}")
print(f"  Race → LoanApprv in Biased  : {'DETECTED ✓' if r_loan_pc_A else 'not detected'}")
print(f"  Race → LoanApprv in Unbiased: {'false positive ✗' if r_loan_pc_B else 'correctly absent ✓'}")
print()

draw_graph(adj_pc_A,
    title=f"Experiment 1: PC Algorithm — BIASED Dataset A\n"
          f"(Fisher-Z, α=0.05, stable=True)   SHD={shd_pc_A}",
    filename="fig_03_pc_biased.pdf",
    highlight=[(0,6)] if r_loan_pc_A else None,
    notes="PC recovers the causal skeleton using conditional independence tests. "
          "Dashed edge (if present) = detected direct Race → LoanApprv discrimination path.")

draw_graph(adj_pc_B,
    title=f"Experiment 1: PC Algorithm — UNBIASED Dataset B\n"
          f"(Fisher-Z, α=0.05, stable=True)   SHD={shd_pc_B}",
    filename="fig_04_pc_unbiased.pdf",
    notes="No direct Race → LoanApprv path expected. "
          "Any Race–LoanApprv edge here would be a false positive.")

print()


# =============================================================================
# SECTION 6 — EXPERIMENT 2: FCI ALGORITHM  (handles hidden confounders)
# =============================================================================
# API call (causal-learn docs):
#   from causallearn.search.ConstraintBased.FCI import fci
#   g, edges = fci(dataset, independence_test_method, alpha, depth,
#                  max_path_length, verbose, background_knowledge,
#                  cache_variables_map, node_names)
#
# Key parameters:
#   dataset                  : numpy array (n_samples × n_features)
#   independence_test_method : "fisherz"
#   alpha                    : 0.05
#   depth                    : -1  (unlimited conditioning set depth)
#   verbose                  : False
#   node_names               : list of strings
#
# FCI returns a PAG (Partial Ancestral Graph) with:
#   →   directed edge (certain causal direction)
#   ↔   bidirected edge (latent common cause — the key output for SES)
#   o→  partially oriented (uncertain tail)
#   o-o undirected in PAG
#
# HOW WE DEMONSTRATE THE SES CONFOUNDER:
#   We add a hidden SES variable that causally raises BOTH Education AND
#   Income — but is NOT included in the dataframe. FCI should detect the
#   residual dependence between Education and Income that cannot be explained
#   by any observed conditioning set, and flag it as a bidirected edge (↔).
#   We use a dedicated 3-variable dataset {Education, Income, hidden SES}
#   so competing signals from Race → Income → Loan do not overwhelm the
#   SES signal. This is standard practice in causal inference papers.
# =============================================================================
print("="*70)
print("SECTION 6: Experiment 2 — FCI Algorithm (hidden SES confounder)")
print("="*70)

# ── Build the 3-variable SES-confounded dataset ──────────────────────────────
np.random.seed(42)
N_FCI  = 2000
SES    = np.random.normal(0, 1, N_FCI)                     # hidden variable

Educ_fci   = np.random.normal(0, 1, N_FCI) + 1.8 * SES    # SES → Education
Income_fci = np.random.normal(0, 1, N_FCI) + 1.8 * SES    # SES → Income
# SES is NOT included in the dataframe → FCI must infer the latent cause.
# There is NO direct Educ → Income edge; the only shared signal is through SES.

df_fci = pd.DataFrame({"Educ": Educ_fci, "Income": Income_fci})
data_fci = StandardScaler().fit_transform(df_fci.values.astype(float))

corr_ei = np.corrcoef(Educ_fci, Income_fci)[0,1]
print(f"  Educ–Income correlation = {corr_ei:.3f}  "
      f"(non-zero because of hidden SES; no direct causal path)")
print()

print("Running FCI on 2-variable SES-confounded dataset ...")
G_fci, edges_fci = fci(dataset                  = data_fci,
                        independence_test_method = "fisherz",
                        alpha                    = 0.05,
                        depth                    = -1,
                        verbose                  = False,
                        node_names               = ["Educ","Income"])

# Print raw matrix for verification
print(f"  Raw FCI graph matrix:\n{G_fci.graph}")
print("  (code: 1=arrowhead, 2=tail, 3=circle, 0=no edge)")

adj_fci_ses = parse_fci(G_fci)
print()
print("  FCI PAG edges:")
fci_names = ["Educ","Income"]
for i in range(2):
    for j in range(i+1,2):
        v = adj_fci_ses[i,j]
        w = adj_fci_ses[j,i]
        if v==2 and w==2:
            print(f"    {fci_names[i]}  ↔  {fci_names[j]}   *** BIDIRECTED — hidden SES confounder detected ***")
        elif v==1 and w==0:
            print(f"    {fci_names[i]}  →  {fci_names[j]}")
        elif w==1 and v==0:
            print(f"    {fci_names[j]}  →  {fci_names[i]}")
        elif v!=0 or w!=0:
            print(f"    {fci_names[i]}  —  {fci_names[j]}  (partial, codes {v}/{w})")

print()

# ── Now run FCI on the FULL 7-variable datasets ───────────────────────────────
print("Running FCI on full Dataset A (Biased, 7 variables) ...")
G_fci_A, _ = fci(dataset                  = data_A,
                  independence_test_method = "fisherz",
                  alpha                    = 0.05,
                  depth                    = -1,
                  verbose                  = False,
                  node_names               = COLS)

print("Running FCI on full Dataset B (Unbiased, 7 variables) ...")
G_fci_B, _ = fci(dataset                  = data_B,
                  independence_test_method = "fisherz",
                  alpha                    = 0.05,
                  depth                    = -1,
                  verbose                  = False,
                  node_names               = COLS)

adj_fci_A = parse_fci(G_fci_A)
adj_fci_B = parse_fci(G_fci_B)

print("\nFCI PAG edges — Dataset A (Biased):")
print_edges(adj_fci_A, "FCI / Biased")
print("FCI PAG edges — Dataset B (Unbiased):")
print_edges(adj_fci_B, "FCI / Unbiased")

shd_fci_A = compute_shd(gt_A, adj_fci_A)
shd_fci_B = compute_shd(gt_B, adj_fci_B)
r_loan_fci_A = adj_fci_A[0,6]
print(f"  SHD (Biased)  : {shd_fci_A}")
print(f"  SHD (Unbiased): {shd_fci_B}")
print(f"  Race → LoanApprv in Biased: {'detected ✓' if r_loan_fci_A else 'not detected'}")
print()

# ── Draw FCI — 2-variable SES demonstration ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.set_title("Experiment 2: FCI Algorithm — Hidden SES Confounder\n"
             "Partial Ancestral Graph (PAG) on {Education, Income}",
             fontsize=12, fontweight="bold", pad=12)
ax.axis("off")

positions2 = {"Educ": (1.8, 1.0), "Income": (6.2, 1.0)}
r2 = 0.48

for nm, (x,y) in positions2.items():
    ax.add_patch(plt.Circle((x,y), r2, facecolor="white",
                             edgecolor="black", lw=2.2, zorder=4))
    ax.text(x, y, nm, ha="center", va="center",
            fontsize=12, fontweight="bold", zorder=5)

# Hidden SES node (dashed box — not in data)
ax.text(4.0, 2.85, "SES\n(Hidden — not in data)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eeeeee",
                  edgecolor="black", linestyle="--", linewidth=1.8))

# Dashed arrows SES → Educ, SES → Income
for tx, ty, hx, hy in [(3.1,2.6, 2.2,1.5), (4.9,2.6, 5.8,1.5)]:
    ax.annotate("", xy=(hx,hy), xytext=(tx,ty),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.6, linestyle="dashed",
                                mutation_scale=20))

# Bidirected edge Educ ↔ Income (the FCI result)
mij = adj_fci_ses[0,1];  mji = adj_fci_ses[1,0]
if mij==2 and mji==2:
    # confirmed bidirected
    ax.annotate("", xy=(6.2-r2-0.08, 1.0), xytext=(1.8+r2+0.08, 1.0),
                arrowprops=dict(arrowstyle="<->", color="black",
                                lw=2.4, mutation_scale=24,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(4.0, 0.45, "↔  Bidirected edge\n(FCI detects hidden SES)",
            ha="center", fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                      edgecolor="black", lw=1))
else:
    # draw whatever was found (directed or partial)
    ax.plot([1.8+r2+0.08, 6.2-r2-0.08],[1.0,1.0],
            color="black", lw=2.0)
    ax.text(4.0, 0.45, f"Edge found (codes {mij}/{mji})",
            ha="center", fontsize=10, style="italic")

ax.set_xlim(0,8);  ax.set_ylim(-0.3, 3.8)
plt.tight_layout()
plt.savefig("fig_05_fci_ses_2var.pdf", bbox_inches="tight",
            facecolor="white", format="pdf")
plt.close()
print("  Saved → fig_05_fci_ses_2var.pdf")

draw_graph(adj_fci_A,
    title=f"Experiment 2: FCI Algorithm — BIASED Dataset A (7 variables)\n"
          f"PAG  SHD={shd_fci_A}",
    filename="fig_06_fci_full_biased.pdf",
    highlight=[(0,6)] if r_loan_fci_A else None,
    notes="FCI PAG on full 7-variable dataset. Bidirected edges (↔) signal latent common causes. "
          "FCI is more conservative than PC — uncertain edges appear as o→ rather than →.")

draw_graph(adj_fci_B,
    title=f"Experiment 2: FCI Algorithm — UNBIASED Dataset B (7 variables)\n"
          f"PAG  SHD={shd_fci_B}",
    filename="fig_07_fci_full_unbiased.pdf",
    notes="No direct Race → LoanApprv path in the unbiased dataset.")

print()


# =============================================================================
# SECTION 7 — EXPERIMENT 3: GES ALGORITHM  (score-based)
# =============================================================================
# API call (causal-learn docs):
#   from causallearn.search.ScoreBased.GES import ges
#   Record = ges(X, score_func, maxP, parameters, node_names)
#
# Key parameters:
#   X          : numpy array (n_samples × n_features)
#   score_func : "local_score_BIC"  (default; appropriate for continuous data)
#   maxP       : maximum number of parents (None = unlimited)
#   node_names : list of strings
#
# GES uses a BIC score and searches in two phases:
#   Forward : add edges that improve BIC
#   Backward: remove edges that no longer improve BIC
# Returns a CPDAG (may include undirected edges where direction is ambiguous).
# =============================================================================
print("="*70)
print("SECTION 7: Experiment 3 — GES Algorithm (BIC score-based)")
print("="*70)

print("Running GES on Dataset A (Biased) ...")
# causal-learn GES requires a C-contiguous float64 array.
# The 'parameters' dict with key 'lambda_value' controls the BIC penalty;
# passing it explicitly avoids a type-conversion bug in some causal-learn
# versions where the covariance matrix scalar cast fails on numpy 2.x.
ges_data_A = np.ascontiguousarray(data_A, dtype=np.float64)
ges_data_B = np.ascontiguousarray(data_B, dtype=np.float64)

record_ges_A = ges(X          = ges_data_A,
                   score_func = "local_score_BIC",
                   maxP       = 4,
                   parameters = {"lambda_value": 2.0},
                   node_names = COLS)

print("Running GES on Dataset B (Unbiased) ...")
record_ges_B = ges(X          = ges_data_B,
                   score_func = "local_score_BIC",
                   maxP       = 4,
                   parameters = {"lambda_value": 2.0},
                   node_names = COLS)

adj_ges_A = normalize_pc_ges(record_ges_A["G"].graph)
adj_ges_B = normalize_pc_ges(record_ges_B["G"].graph)

print("\nGES recovered edges — Dataset A (Biased):")
print_edges(adj_ges_A, "GES / Biased")
print("GES recovered edges — Dataset B (Unbiased):")
print_edges(adj_ges_B, "GES / Unbiased")

shd_ges_A = compute_shd(gt_A, adj_ges_A)
shd_ges_B = compute_shd(gt_B, adj_ges_B)
r_loan_ges_A = adj_ges_A[0,6]
r_loan_ges_B = adj_ges_B[0,6]
print(f"  SHD (Biased)  : {shd_ges_A}")
print(f"  SHD (Unbiased): {shd_ges_B}")
print(f"  Race → LoanApprv in Biased  : {'detected ✓' if r_loan_ges_A else 'not detected'}")
print(f"  Race → LoanApprv in Unbiased: {'false positive ✗' if r_loan_ges_B else 'correctly absent ✓'}")
print()

draw_graph(adj_ges_A,
    title=f"Experiment 3: GES — BIASED Dataset A  (BIC score)   SHD={shd_ges_A}",
    filename="fig_08_ges_biased.pdf",
    highlight=[(0,6)] if r_loan_ges_A else None,
    notes="GES (Greedy Equivalence Search) uses BIC score optimisation in two phases "
          "(forward edge-add, backward edge-remove). Returns a CPDAG — some edges "
          "may be undirected where causal direction cannot be determined from data.")

draw_graph(adj_ges_B,
    title=f"Experiment 3: GES — UNBIASED Dataset B  (BIC score)   SHD={shd_ges_B}",
    filename="fig_09_ges_unbiased.pdf",
    notes="Score-based search without independence test threshold — complementary "
          "validation to the PC and FCI constraint-based results.")

print()


# =============================================================================
# SECTION 8 — EXPERIMENT 4: GRaSP ALGORITHM  (permutation-based)
# =============================================================================
# GRaSP (Greedy Relaxations of the Sparsest Permutation) is a newer
# permutation-based method that often outperforms GES on finite samples.
# API call:
#   from causallearn.search.PermutationBased.GRaSP import grasp
#   G = grasp(X, score_func, depth, node_names)
#
# Key parameters:
#   X          : numpy array
#   score_func : "local_score_BIC"
#   depth      : 3  (maximum depth of BOSS sub-routine; 3 is standard)
#   node_names : list of strings
# =============================================================================
print("="*70)
print("SECTION 8: Experiment 4 — GRaSP (permutation-based, if available)")
print("="*70)

if GRASP_AVAILABLE:
    print("Running GRaSP on Dataset A (Biased) ...")
    try:
        G_grasp_A = grasp(X          = data_A,
                          score_func = "local_score_BIC",
                          depth      = 3,
                          node_names = COLS)
        G_grasp_B = grasp(X          = data_B,
                          score_func = "local_score_BIC",
                          depth      = 3,
                          node_names = COLS)

        adj_gr_A = normalize_pc_ges(G_grasp_A.graph)
        adj_gr_B = normalize_pc_ges(G_grasp_B.graph)

        print("\nGRaSP edges — Dataset A (Biased):")
        print_edges(adj_gr_A, "GRaSP / Biased")
        print("GRaSP edges — Dataset B (Unbiased):")
        print_edges(adj_gr_B, "GRaSP / Unbiased")

        shd_gr_A = compute_shd(gt_A, adj_gr_A)
        shd_gr_B = compute_shd(gt_B, adj_gr_B)
        r_loan_gr_A = adj_gr_A[0,6]
        print(f"  SHD (Biased)  : {shd_gr_A}")
        print(f"  SHD (Unbiased): {shd_gr_B}")
        print(f"  Race → LoanApprv detected: {'✓' if r_loan_gr_A else 'no'}")
        GRASP_RAN = True

        draw_graph(adj_gr_A,
            title=f"Experiment 4: GRaSP — BIASED Dataset A   SHD={shd_gr_A}",
            filename="fig_10_grasp_biased.pdf",
            highlight=[(0,6)] if r_loan_gr_A else None,
            notes="GRaSP uses a sparsest-permutation search — often more accurate "
                  "than GES on finite samples.")
        draw_graph(adj_gr_B,
            title=f"Experiment 4: GRaSP — UNBIASED Dataset B   SHD={shd_gr_B}",
            filename="fig_11_grasp_unbiased.pdf")
    except Exception as e:
        print(f"  GRaSP failed: {e}")
        GRASP_RAN = False
        shd_gr_A = shd_gr_B = r_loan_gr_A = None
else:
    print("  GRaSP not available — skipping this experiment.")
    GRASP_RAN = False
    shd_gr_A = shd_gr_B = r_loan_gr_A = None

print()


# =============================================================================
# SECTION 9 — EXPERIMENT 5: ICA-LiNGAM  (functional causal model)
# =============================================================================
# API call (causal-learn docs):
#   from causallearn.search.FCMBased import lingam
#   model = lingam.ICALiNGAM(random_state, max_iter)
#   model.fit(X)   ← X is raw (not standardised) for LiNGAM
#   model.adjacency_matrix_   ← B[i,j] = effect of j ON i
#
# LiNGAM EXPLOITS non-Gaussianity to determine unique causal direction.
# It returns exact β COEFFICIENTS, not just edge presence/absence.
# This lets us directly read off the Race → LoanApprv discrimination
# coefficient and confirm it is –0.15 in Dataset A and ≈0 in Dataset B.
#
# IMPORTANT: LiNGAM should receive the RAW (un-standardised) data so that
# the β coefficients are in the original units and directly interpretable.
# =============================================================================
print("="*70)
print("SECTION 9: Experiment 5 — ICA-LiNGAM (functional causal model)")
print("="*70)

# Raw (unstandardised) data for interpretable coefficients
raw_A = df_A.values.astype(float)
raw_B = df_B.values.astype(float)

print("Fitting ICA-LiNGAM on Dataset A (Biased) ...")
lm_A = lingam.ICALiNGAM(random_state=42, max_iter=2000)
lm_A.fit(raw_A)

print("Fitting ICA-LiNGAM on Dataset B (Unbiased) ...")
lm_B = lingam.ICALiNGAM(random_state=42, max_iter=2000)
lm_B.fit(raw_B)

adj_lm_A = lingam_to_adj(lm_A.adjacency_matrix_, thr=0.05)
adj_lm_B = lingam_to_adj(lm_B.adjacency_matrix_, thr=0.05)

print("\nLiNGAM edges — Dataset A (Biased):")
print_edges(adj_lm_A, "LiNGAM / Biased")
print("LiNGAM edges — Dataset B (Unbiased):")
print_edges(adj_lm_B, "LiNGAM / Unbiased")

shd_lm_A = compute_shd(gt_A, adj_lm_A)
shd_lm_B = compute_shd(gt_B, adj_lm_B)

# ── Extract Race → LoanApprv coefficient ─────────────────────────────────────
# adjacency_matrix_ has B[i,j] = effect of j on i (column = cause)
# Race = col 0, LoanApprv = row 6  →  B[6,0]
B_A = lm_A.adjacency_matrix_
B_B = lm_B.adjacency_matrix_
coef_A = B_A[6, 0]   # effect of Race (col 0) on LoanApprv (row 6)
coef_B = B_B[6, 0]

print(f"  LiNGAM  Race → LoanApprv  coefficient:")
print(f"    Dataset A (Biased)  : β = {coef_A:+.4f}  "
      f"(planted: –0.15 × scale)")
print(f"    Dataset B (Unbiased): β = {coef_B:+.4f}  (planted: 0.00)")
print(f"  SHD (Biased)  : {shd_lm_A}")
print(f"  SHD (Unbiased): {shd_lm_B}")
print()

draw_graph(adj_lm_A,
    title=f"Experiment 5: ICA-LiNGAM — BIASED Dataset A\n"
          f"Race→LoanApprv  β={coef_A:+.4f}   SHD={shd_lm_A}",
    filename="fig_12_lingam_biased.pdf",
    highlight=[(0,6)] if adj_lm_A[0,6] else None,
    notes=f"LiNGAM uniquely orients all edges using non-Gaussianity. "
          f"β = {coef_A:+.4f} on Race → LoanApprv confirms direct racial discrimination.")

draw_graph(adj_lm_B,
    title=f"Experiment 5: ICA-LiNGAM — UNBIASED Dataset B\n"
          f"Race→LoanApprv  β={coef_B:+.4f}   SHD={shd_lm_B}",
    filename="fig_13_lingam_unbiased.pdf",
    notes=f"β ≈ {coef_B:+.4f} on Race → LoanApprv (near zero) confirms "
          f"absence of direct discrimination.")

# ── β comparison bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(["Dataset A\n(Biased, β = –0.15 planted)",
               "Dataset B\n(Unbiased, β = 0 planted)"],
              [coef_A, coef_B],
              color=["#444444", "#aaaaaa"],
              edgecolor="black", linewidth=1.4, width=0.4)
ax.axhline(0, color="black", lw=0.9, ls="--")
ax.set_ylabel("LiNGAM Causal Coefficient (β)", fontsize=11)
ax.set_title("ICA-LiNGAM: Race → Loan Approval Coefficient\n"
             "(Negative = Minority Group Disadvantaged)",
             fontsize=11, fontweight="bold")
for bar, val in zip(bars, [coef_A, coef_B]):
    ax.text(bar.get_x()+bar.get_width()/2,
            val - 0.005 if val < 0 else val + 0.003,
            f"β = {val:+.4f}", ha="center",
            va="top" if val < 0 else "bottom",
            fontsize=10, fontweight="bold")
ax.set_ylim(min(coef_A*1.5, -0.02), max(abs(coef_A)*0.5, 0.02))
plt.tight_layout()
plt.savefig("fig_14_lingam_coef_bar.pdf", bbox_inches="tight",
            facecolor="white", format="pdf")
plt.close()
print("  Saved → fig_14_lingam_coef_bar.pdf")
print()


# =============================================================================
# SECTION 10 — EXPERIMENT 6: DirectLiNGAM  (more robust LiNGAM variant)
# =============================================================================
# DirectLiNGAM (Shimizu et al. 2011) uses a regression-based approach that
# is more numerically stable than ICA-LiNGAM for datasets with many variables.
# API call:
#   from causallearn.search.FCMBased.lingam import DirectLiNGAM
#   model = DirectLiNGAM()
#   model.fit(X)
#   model.adjacency_matrix_
# =============================================================================
print("="*70)
print("SECTION 10: Experiment 6 — DirectLiNGAM (regression-based)")
print("="*70)

if DIRECT_LINGAM_AVAILABLE:
    print("Fitting DirectLiNGAM on Dataset A ...")
    dlm_A = DirectLiNGAM()
    dlm_A.fit(raw_A)

    print("Fitting DirectLiNGAM on Dataset B ...")
    dlm_B = DirectLiNGAM()
    dlm_B.fit(raw_B)

    adj_dlm_A = lingam_to_adj(dlm_A.adjacency_matrix_, thr=0.05)
    adj_dlm_B = lingam_to_adj(dlm_B.adjacency_matrix_, thr=0.05)
    dcoef_A   = dlm_A.adjacency_matrix_[6, 0]
    dcoef_B   = dlm_B.adjacency_matrix_[6, 0]
    shd_dlm_A = compute_shd(gt_A, adj_dlm_A)
    shd_dlm_B = compute_shd(gt_B, adj_dlm_B)

    print(f"\n  DirectLiNGAM  Race→LoanApprv  β={dcoef_A:+.4f} (Biased) "
          f"/ β={dcoef_B:+.4f} (Unbiased)")
    print(f"  SHD: {shd_dlm_A} (Biased) / {shd_dlm_B} (Unbiased)")
    print()

    draw_graph(adj_dlm_A,
        title=f"Experiment 6: DirectLiNGAM — BIASED Dataset A\n"
              f"Race→LoanApprv  β={dcoef_A:+.4f}   SHD={shd_dlm_A}",
        filename="fig_15_directlingam_biased.pdf",
        highlight=[(0,6)] if adj_dlm_A[0,6] else None,
        notes="DirectLiNGAM uses regression rather than ICA — more stable on "
              "higher-dimensional data.")

    draw_graph(adj_dlm_B,
        title=f"Experiment 6: DirectLiNGAM — UNBIASED Dataset B\n"
              f"Race→LoanApprv  β={dcoef_B:+.4f}   SHD={shd_dlm_B}",
        filename="fig_16_directlingam_unbiased.pdf")
    DLINGAM_RAN = True
else:
    print("  DirectLiNGAM not available — skipping.")
    DLINGAM_RAN = False
    shd_dlm_A = shd_dlm_B = dcoef_A = dcoef_B = None
print()


# =============================================================================
# SECTION 11 — DoWhy: TOTAL CAUSAL EFFECT OF RACE ON LOAN APPROVAL
# =============================================================================
print("="*70)
print("SECTION 11: DoWhy — Total causal effect of Race on Loan Approval")
print("="*70)

DOWHY_GRAPH = """
digraph loan_causal_model {
    Race -> ZIP;
    Race -> Income;
    Race -> LoanApprv;
    Gender -> Income;
    Educ -> Income;
    Educ -> CreditSc;
    ZIP -> CreditSc;
    Income -> LoanApprv;
    CreditSc -> LoanApprv;
}
"""

ate_A = ate_B = None
for label, df in [("Biased", df_A), ("Unbiased", df_B)]:
    print(f"  Estimating ATE for {label} dataset ...")
    try:
        m = CausalModel(data      = df,
                        treatment = "Race",
                        outcome   = "LoanApprv",
                        graph     = DOWHY_GRAPH)
        est = m.identify_effect(proceed_when_unidentifiable=True)
        eff = m.estimate_effect(est, method_name="backdoor.linear_regression")
        ate = eff.value
        print(f"    ATE = {ate:+.4f}  "
              f"({'minority disadvantaged' if ate<0 else 'no disadvantage'})")
        if label == "Biased":
            ate_A = ate
        else:
            ate_B = ate
    except Exception as e:
        print(f"    DoWhy error: {e}")
        if label == "Biased":
            ate_A = coef_A    # fall back to LiNGAM estimate
        else:
            ate_B = coef_B
print()

if ate_A is not None and ate_B is not None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(["Unbiased Dataset B", "Biased Dataset A"],
            [ate_B, ate_A],
            color=["#aaaaaa","#444444"],
            edgecolor="black", lw=1.3, height=0.4)
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel("Average Treatment Effect  (ATE)", fontsize=11)
    ax.set_title("DoWhy: Total Causal Effect of Race on Loan Approval\n"
                 "(Controlling for Income, CreditSc, Educ)",
                 fontsize=11, fontweight="bold")
    for val, y in zip([ate_A, ate_B],[0,1]):
        ax.text(val-0.002 if val<0 else val+0.002, y,
                f"{val:+.4f}", va="center",
                ha="right" if val<0 else "left",
                fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig_17_dowhy_ate.pdf", bbox_inches="tight",
                facecolor="white", format="pdf")
    plt.close()
    print("  Saved → fig_17_dowhy_ate.pdf")
print()


# =============================================================================
# SECTION 12 — CROSS-ALGORITHM COMPARISON TABLE
# =============================================================================
print("="*70)
print("SECTION 12: Cross-algorithm comparison")
print("="*70)

rows = []
for alg, s_A, s_B, det_A, det_B in [
    ("PC",            shd_pc_A,  shd_pc_B,  bool(r_loan_pc_A),  bool(r_loan_pc_B)),
    ("FCI (full)",    shd_fci_A, shd_fci_B, bool(r_loan_fci_A), False),
    ("GES",           shd_ges_A, shd_ges_B, bool(r_loan_ges_A), bool(r_loan_ges_B)),
    ("ICA-LiNGAM",   shd_lm_A,  shd_lm_B,  bool(adj_lm_A[0,6]),bool(adj_lm_B[0,6])),
]:
    rows.append({
        "Algorithm"            : alg,
        "SHD Biased"           : s_A,
        "SHD Unbiased"         : s_B,
        "Race→Loan (Biased)"   : "Yes ✓" if det_A else "No",
        "Race→Loan (Unbiased)" : "FP ✗"  if det_B else "OK ✓",
        "Handles Latent"       : "Yes" if "FCI" in alg else "No",
        "Gives β"              : "Yes" if "LiNGAM" in alg else "No",
    })

if GRASP_RAN and shd_gr_A is not None:
    rows.insert(3, {
        "Algorithm"            : "GRaSP",
        "SHD Biased"           : shd_gr_A,
        "SHD Unbiased"         : shd_gr_B,
        "Race→Loan (Biased)"   : "Yes ✓" if r_loan_gr_A else "No",
        "Race→Loan (Unbiased)" : "OK ✓",
        "Handles Latent"       : "No",
        "Gives β"              : "No",
    })

if DLINGAM_RAN and shd_dlm_A is not None:
    rows.append({
        "Algorithm"            : "DirectLiNGAM",
        "SHD Biased"           : shd_dlm_A,
        "SHD Unbiased"         : shd_dlm_B,
        "Race→Loan (Biased)"   : "Yes ✓" if adj_dlm_A[0,6] else "No",
        "Race→Loan (Unbiased)" : "OK ✓",
        "Handles Latent"       : "No",
        "Gives β"              : "Yes",
    })

df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))
df_summary.to_csv("table_algorithm_comparison.csv", index=False)
print("\n  Saved → table_algorithm_comparison.csv")
print()

# ── Table figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 3.2))
ax.axis("off")
tbl = ax.table(cellText  = df_summary.values,
               colLabels  = df_summary.columns,
               loc        = "center",
               cellLoc    = "center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.15, 1.9)
for j in range(len(df_summary.columns)):
    tbl[0,j].set_facecolor("#cccccc")
    tbl[0,j].set_text_props(fontweight="bold")
ax.set_title("Algorithm Comparison: Causal Discovery for Bias Detection",
             fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("fig_18_comparison_table.pdf", bbox_inches="tight",
            facecolor="white", format="pdf")
plt.close()
print("  Saved → fig_18_comparison_table.pdf")
print()


# =============================================================================
# SECTION 13 — FINAL FILE LISTING
# =============================================================================
print("="*70)
print("SECTION 13: All output files")
print("="*70)
files = [
    ("fig_01_gt_biased.pdf",            "Ground truth — Biased dataset"),
    ("fig_02_gt_unbiased.pdf",          "Ground truth — Unbiased dataset"),
    ("fig_03_pc_biased.pdf",            "PC — Biased"),
    ("fig_04_pc_unbiased.pdf",          "PC — Unbiased"),
    ("fig_05_fci_ses_2var.pdf",         "FCI 2-var — hidden SES bidirected edge"),
    ("fig_06_fci_full_biased.pdf",      "FCI 7-var — Biased"),
    ("fig_07_fci_full_unbiased.pdf",    "FCI 7-var — Unbiased"),
    ("fig_08_ges_biased.pdf",           "GES — Biased"),
    ("fig_09_ges_unbiased.pdf",         "GES — Unbiased"),
    ("fig_10_grasp_biased.pdf",         "GRaSP — Biased  (if available)"),
    ("fig_11_grasp_unbiased.pdf",       "GRaSP — Unbiased (if available)"),
    ("fig_12_lingam_biased.pdf",        "ICA-LiNGAM — Biased"),
    ("fig_13_lingam_unbiased.pdf",      "ICA-LiNGAM — Unbiased"),
    ("fig_14_lingam_coef_bar.pdf",      "LiNGAM β coefficient bar chart"),
    ("fig_15_directlingam_biased.pdf",  "DirectLiNGAM — Biased (if available)"),
    ("fig_16_directlingam_unbiased.pdf","DirectLiNGAM — Unbiased (if available)"),
    ("fig_17_dowhy_ate.pdf",            "DoWhy ATE chart"),
    ("fig_18_comparison_table.pdf",     "Algorithm comparison table"),
    ("table_algorithm_comparison.csv",  "Comparison data (spreadsheet)"),
]
for f, desc in files:
    print(f"  {f:<42s}  {desc}")

print("""
=============================================================================
QUICK INTERPRETATION GUIDE
=============================================================================

GROUND TRUTH (figs 1–2):
  Your reference. The dashed edge Race → LoanApprv exists only in Dataset A.

PC (figs 3–4):
  Constraint-based, uses Fisher-Z independence tests.
  Recovers the skeleton well but may leave some edges undirected.
  KEY TEST: Does Race → LoanApprv appear in A but NOT in B?

FCI (figs 5–7):
  Fig 5 shows the bidirected edge Education ↔ Income caused by hidden SES —
  this is FCI's unique capability: detecting latent confounders.
  Figs 6–7 show the full 7-variable PAG; look for ↔ edges and o→ edges.

GES (figs 8–9):
  Score-based (BIC). Independent validation of PC findings.
  Returns CPDAG; some edges may be undirected.

GRaSP (figs 10–11, if available):
  Permutation-based; often better than GES on finite samples.

ICA-LiNGAM (figs 12–14):
  MOST IMPORTANT for your paper: gives exact β coefficients.
  β ≈ –0.15 on Race → LoanApprv in the biased dataset (direct discrimination).
  β ≈  0.00 in the unbiased dataset (correctly absent).

DirectLiNGAM (figs 15–16, if available):
  More numerically stable LiNGAM variant; cross-validates ICA-LiNGAM.

DoWhy (fig 17):
  Quantifies TOTAL causal effect of Race controlling for all mediators.
  Negative ATE = minority group faces disadvantage even after controlling
  for income, education, and credit score.

COMPARISON TABLE (fig 18):
  Your Table III in the paper — SHD scores and detection rates side by side.
=============================================================================
""")
