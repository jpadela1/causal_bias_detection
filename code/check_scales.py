import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("data/compas-scores-two-years.csv")   # <-- keep your actual filename

print("Columns:", list(df.columns))
print("Race values:", df["race"].value_counts().to_dict())

# Restrict to the two groups and encode: African-American=1, Caucasian=0
df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
df["race_bin"] = (df["race"] == "African-American").astype(int)
print("n after restriction:", len(df))          # expect 5,278

s = df["decile_score"]
print("\nSD of raw Score:", s.std())
raw_gap = s[df.race_bin == 1].mean() - s[df.race_bin == 0].mean()
print("Raw mean gap:", raw_gap)                 # expect ~1.64
print("Gap in SD units:", raw_gap / s.std())

print("\nraw-scale coef:  ",
      smf.ols("decile_score ~ race_bin", df).fit().params["race_bin"])

zs = (s - s.mean()) / s.std()
d2 = df.assign(z_score=zs)
print("std-outcome coef:",
      smf.ols("z_score ~ race_bin", d2).fit().params["race_bin"])