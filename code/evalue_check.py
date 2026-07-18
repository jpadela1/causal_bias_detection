"""
evalue_check.py — VanderWeele & Ding (2017) E-value for the COMPAS
adjusted direct effect. Continuous-outcome approximation:
d -> RR via RR ~= exp(0.91 * d); E = RR + sqrt(RR*(RR-1)).
"""
import numpy as np

SD_SCORE = 2.856          # SD of decile_score on the n=5,278 frame
DE_POINT = 0.5255          # adjusted direct effect, decile points
DE_CI_LO = 0.395           # lower 95% bootstrap CI bound

def evalue_from_effect(effect_points: float, sd: float) -> tuple[float, float]:
    d  = effect_points / sd
    rr = np.exp(0.91 * d)
    return rr, rr + np.sqrt(rr * (rr - 1))

for label, eff in [("point estimate", DE_POINT), ("CI lower bound", DE_CI_LO)]:
    rr, ev = evalue_from_effect(eff, SD_SCORE)
    print(f"{label:15s} DE={eff:+.4f}  d={eff/SD_SCORE:.4f}  RR={rr:.4f}  E-value={ev:.4f}")