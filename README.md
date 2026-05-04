# Causal-Discovery Audit for Algorithmic Bias

Python/PyCharm accompanies the paper *"From Correlation to Mechanism: A
Causal-Discovery Protocol for Auditing Algorithmic Bias in High-Stakes
Decision Systems."* Implements all six causal discovery algorithms, the
synthetic loan-approval study, the sensitivity sweep, the COMPAS
application, and Pearl-style backdoor ATE estimation.

## Project layout

```
causal_bias_audit/
├── requirements.txt
├── synthetic_data.py         # SCM-based data generation (Eqs. in §IV-A)
├── causal_discovery.py       # PC, FCI, GES, GRaSP, ICA-LiNGAM, DirectLiNGAM
├── ate_estimation.py         # Backdoor ATE via OLS (Eq. 1-3)
├── visualization.py          # DAG plotting (matches the paper's color scheme)
├── sensitivity_analysis.py   # Sweep over beta and n (Section V)
├── compas_analysis.py        # COMPAS load + ProPublica preprocessing
├── main_synthetic.py         # Run Study 1 (Section IV)
├── main_sensitivity.py       # Run Study 1.5 (Section V)
├── main_compas.py            # Run Study 2 (Section VI)
├── data/                     # COMPAS CSV cached here on first run
├── figures/                  # Plots written here
└── results/                  # CSV summary tables written here
```

## PyCharm setup

1. **Open the folder** `causal_bias_audit/` in PyCharm (`File > Open`).
2. **Create a virtualenv interpreter**: `File > Settings > Project >
   Python Interpreter > Add Interpreter > Add Local Interpreter > Virtualenv`.
   Use Python 3.10 or 3.11.
3. **Install dependencies**: open the built-in terminal (`Alt+F12`) and run
   ```bash
   pip install -r requirements.txt
   ```
   (`causal-learn` will pull `numpy`, `scipy`, `scikit-learn`, etc.; the
   first install can take a few minutes.)
4. **Mark the project root as Sources Root** so the cross-file imports
   resolve cleanly: right-click the `causal_bias_audit/` folder in the
   Project pane > `Mark Directory as > Sources Root`.
5. **Run configurations**: right-click any `main_*.py` file > `Run` to
   generate a configuration. PyCharm will also let you set
   `Working directory` to the project root (recommended so that the
   `data/`, `figures/`, and `results/` folders land in the right place).

## How to run

Run the three entry points in order. Each is a normal Python script that
writes CSV summaries to `results/` and PNG figures to `figures/`.

```bash
python main_synthetic.py     # Study 1 - ~30 seconds
python main_sensitivity.py   # Section V - several minutes (small grid)
python main_compas.py        # Study 2 - ~1 minute (downloads ~5 MB CSV)
```

To run the **full** sensitivity grid that the paper reports
(`betas = [0, 0.05, 0.10, 0.15, 0.20, 0.25]`,
`n in {1000, 50000}`, `n_repeats = 20`), edit `main_sensitivity.py` and
expand the lists. Expect 1+ hours on a laptop.

## What each script reproduces

### `main_synthetic.py` -> Section IV / Table II
- Generates Dataset A (biased, β = -0.15) and Dataset B (unbiased, β = 0)
  using the SCM from §IV-A with `np.random.seed`-equivalent seed=42.
- Runs all six algorithms.
- Prints whether each algorithm detects the planted `Race -> Loan` edge
  and reports SHD against ground truth. The two LiNGAM variants additionally
  return a numerical β̂ for that edge.
- Saves one DAG per algorithm-dataset combination plus a 2x3 grid figure.
- Computes backdoor-adjusted ATE for `Race -> Loan` over progressively
  larger adjustment sets.

### `main_sensitivity.py` -> Section V / Figure 2
- Sweeps over (β, n, seed) and records detection rate + SHD per
  algorithm-cell.
- Aggregates and writes `results/sensitivity_summary.csv`.
- Draws a 2-row bar-grid figure analogous to the paper's Figure 2: bar
  height = detection rate, annotation = mean SHD ± std.

### `main_compas.py` -> Section VI / Table III + Figures 3-5
- Downloads the ProPublica COMPAS CSV (cached after first run).
- Applies the standard ProPublica filters
  (`|days_b_screening_arrest| <= 30`, `is_recid != -1`, valid charge
  degree, valid score) and encodes the nine analysis variables.
- Runs all six algorithms; tabulates `Race -> Score` and
  `Race -> Priors` detections + LiNGAM β̂.
- Computes backdoor-adjusted ATE in three stages
  (no controls -> age+charge -> full criminal history), reproducing
  Equations (1)-(3) in the paper.
- Saves a DAG per algorithm + a grid figure.

## ATE: formula and implementation

The paper uses Pearl's backdoor adjustment:

```
ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
    = E_Z[ E[Y | T=1, Z] - E[Y | T=0, Z] ]
```

For an admissible adjustment set Z (non-descendants of T that block all
backdoor paths from T to Y), under a linear outcome model
`Y = β₀ + β_T·T + β_Z·Z + ε` the ATE is identified by the OLS
coefficient on T:

```python
ATE_hat = β_T_hat   # the coefficient on T after adjusting for Z
```

This is implemented in `ate_estimation.backdoor_ate`. The function
`staged_backdoor_ate` runs the procedure for a list of progressively
richer adjustment sets to reproduce the paper's three-row ATE tables.

If you prefer the DoWhy implementation the paper cites, install
`dowhy>=0.11` and add a `do_why` branch to `backdoor_ate`. Both
estimators agree on these datasets to four decimal places because the
underlying assumption (linear no-interaction outcome) is the same; OLS
is just the closed-form estimator.

## Notes on edge-encoding and SHD - CHECK

- `causal_discovery._extract_edges_from_graph` converts causal-learn's
  graph matrix to lists of directed / undirected / bidirected edges.
- For LiNGAM, `B[i, j] != 0` encodes `j -> i` with coefficient `B[i, j]`;
  `DiscoveryResult.get_coefficient(src, dst)` is the safe accessor.
- `structural_hamming_distance` counts: extra edges + missing edges +
  reversed orientations + undirected predictions where the truth is
  directed.

## Known limitation: GRaSP

GRaSP can produce causally-incoherent edges (e.g. directing arrows
INTO `Race`) on datasets where the outcome variable is highly
multicollinear with its inputs (the case for any system that scores
people on their own features). The sensitivity analysis confirms a
near-zero detection rate regardless of β or n. The COMPAS run keeps
GRaSP enabled but the paper recommends excluding it from production
audits of algorithmic-score systems.

## Reproducibility

- All RNG seeds are explicit. Synthetic-data seed defaults to 42 to
  match the paper.
