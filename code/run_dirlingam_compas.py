from compas_analysis import load_compas, preprocess_compas
from causal_discovery import run_direct_lingam

df = preprocess_compas(load_compas(), restrict_to_aa_caucasian=True)
res = run_direct_lingam(df)   # no exogenous/sinks kwargs = unconstrained
print("Race->Score:", res.has_directed_edge("Race", "Score"),
      "beta:", res.get_coefficient("Race", "Score"))
print("Edges into Race:", [(s,d) for (s,d) in res.directed_edges if d=="Race"])
print(res.summary())