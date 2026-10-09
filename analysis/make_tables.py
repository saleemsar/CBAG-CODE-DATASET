"""Emit LaTeX fragments (data-heavy tables and plot coordinates) straight from the CSVs."""
import os
import pandas as pd, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
T = {"UAV1": f"{R}/UAV1", "UAV2": f"{R}/UAV2_AccessPoint"}
NAME = {"BruteForce": "Brute-Force", "FakeLanding": "Fake-Landing", "replay": "Replay", "Fake Landing": "Fake-Landing"}
tt = lambda f: "\\texttt{" + f.replace("_", "\\_") + "}"
out = {}

# DIF quadrant coordinates
d = pd.read_csv(f"{R}/UAV1/DIFS_scores.csv")
q = []
for quad in ["Core", "Vulnerable", "Ancillary", "Noise"]:
    s = d[d.quadrant == quad]
    q.append(f"% {quad} ({len(s)})\n" + " ".join(f"({a:.3f},{b:.3f})" for a, b in zip(s.MPI_norm, s.FPC_norm)))
out["quad"] = "\n".join(q) + f"\n% medians MPI={d.MPI_norm.median():.3f} FPC={d.FPC_norm.median():.3f}"


# per-feature displacement matrix (operating survivor set, z-score units)
def disp(t, order):
    t9 = pd.read_csv(f"{T[t]}/T9_feature_perturbation.csv")
    n = t9.groupby("class").n_survivors.first()
    feats = (t9.groupby("feature").frac_moved.mean().sort_values(ascending=False).index.tolist())
    rows = []
    for f in feats:
        cells = []
        for c in order:
            r = t9[(t9["class"] == c) & (t9.feature == f)]
            if r.empty:
                cells.append("---")
            else:
                r = r.iloc[0]
                v = f"{r.mean_abs_delta:.2f}" if r.mean_abs_delta >= 0.005 else f"{r.mean_abs_delta:.3f}"
                if r.frac_moved >= 0.995:
                    cells.append(v)
                else:
                    cells.append(f"{v}\\,({r.frac_moved:.2f})".replace("(0.", "(."))
        rows.append(tt(f) + " & " + " & ".join(cells) + "\\\\")
    head = " & ".join(f"{NAME.get(c, c)} ({n[c]:,})".replace(",", "{,}") for c in order)
    return head, "\n".join(rows)


out["disp1"] = disp("UAV1", ["De-authentication", "DoS", "replay", "ICMP Flooding", "Scanning", "MITM", "BruteForce"])
out["disp2"] = disp("UAV2", ["Fake Landing", "Scanning"])


# full ranges table
def rng(t):
    a = pd.read_csv(f"{T[t]}/A6_ranges.csv")
    lines = []
    for c, g in a.groupby("class", sort=False):
        if (g.rho == 0).all() and (g.surv_lo == g.surv_hi).all():
            lines.append(f"{NAME.get(c, c)} & ({len(g)} features) & --- & single point & 0.000 & ---\\\\")
            continue
        g = g.sort_values("rho")
        for i, r in enumerate(g.itertuples()):
            fmt = lambda x: ("0.000" if abs(x) < 5e-4 else f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}" if abs(x) >= 10 else f"{x:.3f}").replace(",", "{,}")
            tight = f"{r.tightening_x:.1f}$\\times$" if r.rho > 0 else "---"
            lab = f"\\multirow{{{len(g)}}}{{*}}{{{NAME.get(c, c)}}}" if i == 0 else ""
            lines.append(f"{lab} & {tt(r.feature)} & $[{fmt(r.raw_lo)},\\,{fmt(r.raw_hi)}]$ & "
                         f"$[{fmt(r.surv_lo)},\\,{fmt(r.surv_hi)}]$ & {r.rho:.3f} & {tight}\\\\")
        lines.append("\\addlinespace")
    return "\n".join(lines)


out["rng1"] = rng("UAV1"); out["rng2"] = rng("UAV2")

# per-class strict sweeps
for t in ["UAV1", "UAV2"]:
    sw = pd.read_csv(f"{T[t]}/gate3_L1_sweep.csv")
    cols = [x for x in sw.columns if x.startswith("FVAS<")]
    lines = []
    for key in ["gen", "class"]:
        s = sw.groupby(key)[cols].sum()
        s = s.sort_values(cols[-1], ascending=False)
        for idx, r in s.iterrows():
            vals = " & ".join(f"{int(v):,}".replace(",", "{,}") for v in r.values)
            lines.append(f"{NAME.get(idx, idx)} & {vals}\\\\")
        lines.append("\\midrule")
    out[f"sw_{t}"] = "\n".join(lines)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fragments.txt"), "w") as fh:
    for k, v in out.items():
        fh.write(f"===== {k}\n{v if isinstance(v, str) else v[0] + chr(10) + v[1]}\n")
print(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fragments.txt")).read())
