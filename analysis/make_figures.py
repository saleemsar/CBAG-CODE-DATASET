"""Regenerate the two data figures of the rebuilt Results section from the run CSVs."""
import os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)
TOP = {"UAV1": f"{R}/UAV1", "UAV2": f"{R}/UAV2_AccessPoint"}
CAPPED = {"C-BO", "HopSkipJump", "Sign-OPT"}
GENS = ["C-SA", "C-PSO", "C-GA", "C-RL", "CG-Diffusion", "CC-CGAN", "C-BO", "HopSkipJump", "Sign-OPT"]
GLAB = ["C-SA", "C-PSO", "C-GA", "C-RL", "CG-Diff.", "CC-CGAN", "C-BO$^\\dagger$", "HSJ$^\\dagger$", "Sign-OPT$^\\dagger$"]
NAME = {"BruteForce": "Brute-Force", "FakeLanding": "Fake-Landing", "replay": "Replay",
        "Fake Landing": "Fake-Landing"}

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "pdf.fonttype": 42})

# ---------------- Figure: class x generator heatmap ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})
for ax, t in zip(axes, ["UAV1", "UAV2"]):
    pc = pd.read_csv(f"{TOP[t]}/perclass_gate.csv")
    pc["att"] = np.where(pc.gen.isin(CAPPED), np.minimum(pc.n, 60), pc.n)
    cls_order = (pc.groupby("class").agg(fs=("FVAS", "sum"), ev=("G1_evade", "sum"))
                 .sort_values(["fs", "ev"], ascending=False).index.tolist())
    E = np.zeros((len(cls_order), len(GENS))); F = np.zeros_like(E, dtype=int)
    for i, c in enumerate(cls_order):
        for j, g in enumerate(GENS):
            r = pc[(pc["class"] == c) & (pc.gen == g)].iloc[0]
            E[i, j] = 100 * r.G1_evade / r.att
            F[i, j] = r.FVAS
    im = ax.imshow(E, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if F[i, j] > 0:
                ax.text(j, i, f"{F[i, j]}", ha="center", va="center", fontsize=7.5, fontweight="bold",
                        color="white" if E[i, j] > 55 else INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="none", ec="white" if E[i, j] > 55 else INK, lw=0.6))
    ax.set_xticks(range(len(GENS))); ax.set_xticklabels(GLAB, rotation=55, ha="right")
    ax.set_yticks(range(len(cls_order))); ax.set_yticklabels([NAME.get(c, c) for c in cls_order])
    ax.set_xticks(np.arange(-.5, len(GENS)), minor=True); ax.set_yticks(np.arange(-.5, len(cls_order)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    tau = 0.25 if t == "UAV1" else 0.1
    ax.set_title(f"({'a' if t=='UAV1' else 'b'}) {t}, operating budget $\\tau={tau}$", fontsize=9, color=INK)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("Gate-1 evasion (% of attacked candidates)", color=INK2); cb.outline.set_visible(False)
pass  # heatmap now built as 2x2 in candidate_figures.py
plt.close(fig)

# ---------------- Figure: per-feature tightening ratio ----------------
panels = [("UAV1", "ICMP Flooding", 236), ("UAV1", "Scanning", 28),
          ("UAV2", "Fake Landing", 60), ("UAV2", "Scanning", 137)]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
axes = axes.ravel()
for ax, (t, c, n) in zip(axes, panels):
    a = pd.read_csv(f"{TOP[t]}/A6_ranges.csv")
    a = a[a["class"] == c].sort_values("rho", ascending=False)
    y = np.arange(len(a))
    ax.hlines(y, 0, a.rho, color=GRID, lw=2)
    ax.plot(a.rho, y, "o", ms=6, color="#2a78d6", mec="white", mew=1.2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(y); ax.set_yticklabels(a.feature, fontsize=6.8, family="DejaVu Sans Mono")
    ax.set_xlim(-0.03, 1.05); ax.set_xticks([0, 0.5, 1])
    ax.set_title(f"{t}: {NAME.get(c, c)}\n($n={n}$, median $\\rho={a.rho.median():.3f}$)", fontsize=8)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.supxlabel("tightening ratio $\\rho$ = survivor range / rulebook interval", fontsize=8, color=INK2, y=0.02)
fig.tight_layout(w_pad=1.0, h_pad=1.2)
fig.savefig(f"{OUT}/res_tightening.pdf", bbox_inches="tight")
print("ok")
