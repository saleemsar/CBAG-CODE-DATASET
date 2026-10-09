"""Regenerate the class-by-generator heatmap and the generator budget-sweep figures from the run CSVs."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
PREV = os.path.join(OUT, "previews")
os.makedirs(OUT, exist_ok=True); os.makedirs(PREV, exist_ok=True)
T = {"UAV1": f"{R}/UAV1", "UAV2": f"{R}/UAV2_AccessPoint", "UAV3": f"{R}/UAV3_GCS"}
CAPPED = {"C-BO", "HopSkipJump", "Sign-OPT"}
GENS = ["C-SA", "C-PSO", "C-GA", "C-RL", "CG-Diffusion", "CC-CGAN", "C-BO", "HopSkipJump", "Sign-OPT"]
GLAB = ["C-SA", "C-PSO", "C-GA", "C-RL", "CG-Diff.", "CC-CGAN", "C-BO$^\\dagger$", "HSJ$^\\dagger$", "Sign-OPT$^\\dagger$"]
NAME = {"BruteForce": "Brute-Force", "Bruteforce": "Brute-Force", "FakeLanding": "Fake-Landing", "replay": "Replay",
        "Fake Landing": "Fake-Landing (b)", "Fake Landing ": "Fake-Landing (a)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "pdf.fonttype": 42})


def grid_ax(ax, nr, nc):
    ax.set_xticks(np.arange(-.5, nc), minor=True); ax.set_yticks(np.arange(-.5, nr), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)


# ---- A: strict survivors per (class, generator) at the operating budget vs tau = 2.0
fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw={"wspace": 0.6})
for ax, t, (c_op, c_hi) in zip(axes, ["UAV1", "UAV2"], [("FVAS<0.25", "FVAS<2.0"), ("FVAS<0.1", "FVAS<2.0")]):
    sw = pd.read_csv(f"{T[t]}/gate3_L1_sweep.csv"); sw["class"] = sw["class"].astype(str)
    hi = sw.pivot(index="class", columns="gen", values=c_hi).reindex(columns=GENS)
    op = sw.pivot(index="class", columns="gen", values=c_op).reindex(columns=GENS)
    order = hi.sum(axis=1).sort_values(ascending=False).index
    hi, op = hi.loc[order], op.loc[order]
    M = hi.values.astype(float)
    im = ax.imshow(np.where(M > 0, M, np.nan), cmap=cmap, norm=LogNorm(1, 10000), aspect="auto")
    ax.imshow(np.where(M == 0, 1, np.nan), cmap=LinearSegmentedColormap.from_list("z", ["#f4f3f0", "#f4f3f0"]), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                v = int(M[i, j]); txt = f"{v/1000:.1f}k" if v >= 1000 else str(v)
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 300 else INK,
                        fontweight="bold" if op.values[i, j] > 0 else "normal")
    ax.set_xticks(range(len(GENS))); ax.set_xticklabels(GLAB, rotation=55, ha="right")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME.get(c, c).replace(" (b)", "") for c in order])
    grid_ax(ax, len(order), len(GENS))
    ax.set_title(f"({'c' if t=='UAV1' else 'd'}) {t}, strict survivors at $\\tau=2.0$", fontsize=9)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.set_label("strict survivors (log scale)", color=INK2); cb.outline.set_visible(False)
fig.savefig(f"{OUT}/res_heat_tau2.pdf", bbox_inches="tight"); fig.savefig(f"{PREV}/A_heat_tau2.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---- C: per-generator strict survivors vs budget
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
FULL = ["C-SA", "C-PSO", "C-GA", "C-RL", "CG-Diffusion", "CC-CGAN"]
for ax, t in zip(axes, ["UAV1", "UAV2"]):
    sw = pd.read_csv(f"{T[t]}/gate3_L1_sweep.csv")
    cols = [c for c in sw.columns if c.startswith("FVAS<")]
    taus = [float(c.split("<")[1]) for c in cols]
    g = sw.groupby("gen")[cols].sum()
    for k, gen in enumerate(FULL):
        y = g.loc[gen].values.astype(float)
        ax.plot(taus, np.where(y > 0, y, np.nan), marker="o", ms=4, lw=2, color=CAT[k], label=gen)
    ax.set_yscale("log"); ax.set_xticks(taus); ax.set_xticklabels([f"{x:g}" for x in taus])
    ax.set_xlabel("Gate-3 budget $\\tau$"); ax.set_title(f"({'c' if t=='UAV1' else 'd'}) {t}, strict survivors by generator", fontsize=9)
    ax.axvline(0.25 if t == "UAV1" else 0.1, color=INK2, ls=(0, (2, 3)), lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.5); [ax.spines[s].set_visible(False) for s in ["top", "right"]]
axes[0].set_ylabel("strict survivors (log)")
axes[1].legend(frameon=False, fontsize=7, loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/res_gen_sweep.pdf", bbox_inches="tight"); fig.savefig(f"{PREV}/C_gen_sweep.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---- B: UAV3 class x generator (evasion shading, survivors boxed)
pc = pd.read_csv(f"{T['UAV3']}/perclass_gate.csv"); pc["class"] = pc["class"].astype(str)
pc["att"] = np.where(pc.gen.isin(CAPPED), np.minimum(pc.n, 60), pc.n)
order = pc.groupby("class").agg(fs=("FVAS", "sum"), ev=("G1_evade", "sum")).sort_values(["fs", "ev"], ascending=False).index
E = np.array([[100 * pc[(pc["class"] == c) & (pc.gen == g)].G1_evade.iloc[0] / pc[(pc["class"] == c) & (pc.gen == g)].att.iloc[0] for g in GENS] for c in order])
F = np.array([[pc[(pc["class"] == c) & (pc.gen == g)].FVAS.iloc[0] for g in GENS] for c in order])
fig, ax = plt.subplots(figsize=(4.6, 3.8))
im = ax.imshow(E, cmap=cmap, vmin=0, vmax=100, aspect="auto")
for i in range(E.shape[0]):
    for j in range(E.shape[1]):
        if F[i, j] > 0:
            ax.text(j, i, str(F[i, j]), ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color="white" if E[i, j] > 55 else INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="none", ec="white" if E[i, j] > 55 else INK, lw=0.5))
ax.set_xticks(range(len(GENS))); ax.set_xticklabels(GLAB, rotation=55, ha="right")
ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME.get(c, c) for c in order])
grid_ax(ax, len(order), len(GENS)); ax.set_title("UAV3 ($\\tau=10$, benign mimicry)", fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Gate-1 evasion (% of attacked)", color=INK2); cb.outline.set_visible(False)
fig.savefig(f"{PREV}/B_uav3_heat.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("ok")

# ---- combined 2x2 class x generator figure (replaces res_class_gen_heatmap.pdf + res_heat_tau2.pdf)
fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.4), gridspec_kw={"wspace": 0.62, "hspace": 0.45})
for col, (t, tau, c_op) in enumerate([("UAV1", 0.25, "FVAS<0.25"), ("UAV2", 0.1, "FVAS<0.1")]):
    pc = pd.read_csv(f"{T[t]}/perclass_gate.csv"); pc["class"] = pc["class"].astype(str)
    pc["att"] = np.where(pc.gen.isin(CAPPED), np.minimum(pc.n, 60), pc.n)
    sw = pd.read_csv(f"{T[t]}/gate3_L1_sweep.csv"); sw["class"] = sw["class"].astype(str)
    order = (pc.groupby("class").agg(fs=("FVAS", "sum"), ev=("G1_evade", "sum"))
             .sort_values(["fs", "ev"], ascending=False).index.tolist())
    E = np.array([[100 * pc[(pc["class"] == c) & (pc.gen == g)].G1_evade.iloc[0] / pc[(pc["class"] == c) & (pc.gen == g)].att.iloc[0] for g in GENS] for c in order])
    F = np.array([[pc[(pc["class"] == c) & (pc.gen == g)].FVAS.iloc[0] for g in GENS] for c in order])
    H = sw.pivot(index="class", columns="gen", values="FVAS<2.0").reindex(index=order, columns=GENS).values.astype(float)
    # top: evasion + boxed strict survivors at operating budget
    ax = axes[0, col]
    imE = ax.imshow(E, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if F[i, j] > 0:
                ax.text(j, i, str(F[i, j]), ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white" if E[i, j] > 55 else INK,
                        bbox=dict(boxstyle="round,pad=0.12", fc="none", ec="white" if E[i, j] > 55 else INK, lw=0.5))
    ax.set_title(f"({'a' if col == 0 else 'b'}) {t}, operating budget $\\tau={tau}$", fontsize=9)
    # bottom: strict survivors at tau = 2.0
    ax2 = axes[1, col]
    ax2.imshow(np.where(H == 0, 1, np.nan), cmap=LinearSegmentedColormap.from_list("z", ["#f4f3f0", "#f4f3f0"]), aspect="auto")
    imH = ax2.imshow(np.where(H > 0, H, np.nan), cmap=cmap, norm=LogNorm(1, 10000), aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i, j] > 0:
                v = int(H[i, j]); txt = f"{v/1000:.1f}k" if v >= 1000 else str(v)
                ax2.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                         color="white" if H[i, j] > 300 else INK, fontweight="bold" if F[i, j] > 0 else "normal")
    ax2.set_title(f"({'c' if col == 0 else 'd'}) {t}, strict survivors at $\\tau=2.0$", fontsize=9)
    for a in (ax, ax2):
        a.set_xticks(range(len(GENS))); a.set_xticklabels(GLAB, rotation=55, ha="right")
        a.set_yticks(range(len(order))); a.set_yticklabels([NAME.get(c, c).replace(" (b)", "") for c in order])
        grid_ax(a, len(order), len(GENS))
cb1 = fig.colorbar(imE, ax=axes[0, :], fraction=0.025, pad=0.02); cb1.set_label("Gate-1 evasion (% of attacked)", color=INK2); cb1.outline.set_visible(False)
cb2 = fig.colorbar(imH, ax=axes[1, :], fraction=0.025, pad=0.02); cb2.set_label("strict survivors (log scale)", color=INK2); cb2.outline.set_visible(False)
fig.savefig(f"{OUT}/res_class_gen_heatmap.pdf", bbox_inches="tight"); fig.savefig(f"{PREV}/heat_2x2.png", dpi=100, bbox_inches="tight"); plt.close(fig)
print("combined ok")
