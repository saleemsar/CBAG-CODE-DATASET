"""Recompute every number used in the rebuilt Results section from the run CSVs.
No value is estimated; everything is derived from results/*/ files."""
import os
import pandas as pd, numpy as np
from scipy.stats import beta, norm

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TOP = {"UAV1": f"{R}/UAV1", "UAV2": f"{R}/UAV2_AccessPoint", "UAV3": f"{R}/UAV3_GCS"}
CAPPED = {"C-BO", "HopSkipJump", "Sign-OPT"}
QUERIES = {"C-SA": 400, "C-PSO": 480, "C-GA": 480, "C-RL": 240, "C-BO": 28,
           "CG-Diffusion": 655, "CC-CGAN": 75, "HopSkipJump": 312, "Sign-OPT": 240}
TAU = {"UAV1": 0.25, "UAV2": 0.1, "UAV3": 10}


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def load(t):
    pc = pd.read_csv(f"{TOP[t]}/perclass_gate.csv")
    pc["class"] = pc["class"].astype(str)
    pc["att"] = np.where(pc.gen.isin(CAPPED), np.minimum(pc.n, 60), pc.n)
    t7 = pd.read_csv(f"{TOP[t]}/T7_constrained_vs_baseline.csv").set_index("gen")
    return pc, t7


out = {}
for t in ["UAV1", "UAV2", "UAV3"]:
    pc, t7 = load(t)
    print(f"\n==================== {t}  (tau_oper={TAU[t]})")
    att = pc.att.sum(); ev = pc.G1_evade.sum(); strict_func = pc.G1G2_func.sum(); fstrict = pc.FVAS.sum()
    orc = t7.functional.sum(); foper = t7.FVAS.sum()
    print(f"attacked {att}  evaded {ev} ({100*ev/att:.1f}%)")
    print(f"oracle-valid evaders {orc} ({100*orc/ev:.1f}% of evaders; G2 removes {100*(1-orc/ev):.1f}%)")
    print(f"oracle+CMR evaders {strict_func} ({100*strict_func/ev:.1f}% of evaders; removes {100*(1-strict_func/ev):.1f}%)")
    print(f"FVAS_oper {foper} (ret {100*foper/att:.3f}%; G3 removes {100*(1-foper/orc):.1f}% of oracle-valid)")
    print(f"FVAS_strict {fstrict} (ret {100*fstrict/att:.3f}%; G3 removes {100*(1-fstrict/strict_func):.1f}% of strict-func)")
    if foper:
        print(f"CMR rejection at oper tau {100*(1-fstrict/foper):.1f}%")

    g = pc.groupby("gen").agg(att=("att", "sum"), ev=("G1_evade", "sum"), sf=("G1G2_func", "sum"), fs=("FVAS", "sum"))
    g["orc"] = t7.functional; g["foper"] = t7.FVAS
    g["ev%"] = 100 * g.ev / g.att
    g["FVR%"] = 100 * g.orc / g.ev.replace(0, np.nan)
    g["q"] = [QUERIES[x] for x in g.index]
    g["eff_oper"] = 1000 * g.foper / (g.att * g.q)
    g["eff_strict"] = 1000 * g.fs / (g.att * g.q)
    ci = [cp(int(k), int(n)) for k, n in zip(g.fs, g.att)]
    g["ret%"] = 100 * g.fs / g.att
    g["ci_lo"] = [100 * a for a, b in ci]; g["ci_hi"] = [100 * b for a, b in ci]
    pd.set_option("display.width", 250)
    print(g.round(3).to_string())

    # z-test top two strict
    top = g.sort_values("fs", ascending=False).head(2)
    (a, b) = top.index
    k1, n1, k2, n2 = g.loc[a, "fs"], g.loc[a, "att"], g.loc[b, "fs"], g.loc[b, "att"]
    p = (k1 + k2) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se
    print(f"z-test {a} vs {b}: z={z:.2f}, p={2*norm.sf(abs(z)):.2e}")

    c = pc.groupby("class").agg(att=("att", "sum"), ev=("G1_evade", "sum"), sf=("G1G2_func", "sum"), fs=("FVAS", "sum"))
    c["ASR%"] = 100 * c.ev / c.att
    try:
        t9 = pd.read_csv(f"{TOP[t]}/T9_feature_perturbation.csv")
        c["oper"] = t9.groupby("class").n_survivors.first()
    except Exception:
        pass
    print(c.sort_values("fs", ascending=False).round(1).to_string())
    top2 = c.fs.sort_values(ascending=False).head(2).sum()
    print(f"top-2 class share of strict: {100*top2/max(fstrict,1):.1f}%  (classes w/ strict>0: {(c.fs>0).sum()})")

    sw = pd.read_csv(f"{TOP[t]}/gate3_L1_sweep.csv")
    sw["class"] = sw["class"].astype(str)
    cols = [x for x in sw.columns if x.startswith("FVAS<")]
    print("per-generator strict sweep:\n", sw.groupby("gen")[cols].sum().to_string())
    print("per-class strict sweep:\n", sw.groupby("class")[cols].sum().to_string())
    print("total:", sw[cols].sum().to_dict())
    t3 = pd.read_csv(f"{TOP[t]}/T3_reachability.csv")
    t3["cmr_rej%"] = 100 * (1 - t3.FVAS_strict_CMR / t3.FVAS_operating)
    print(t3.round(1).to_string())
    out[t] = dict(pc=pc, g=g, c=c)

# DIF medians
d = pd.read_csv(f"{R}/UAV1/DIFS_scores.csv")
print("\nDIF medians MPI_norm %.3f FPC_norm %.3f" % (d.MPI_norm.median(), d.FPC_norm.median()))
print(d.quadrant.value_counts().to_dict())
