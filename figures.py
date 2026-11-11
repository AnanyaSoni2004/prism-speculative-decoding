"""Figures and tables. Reads only results/*.csv, *.json and *.npz; writes results/figures/ and results/table*.{csv,md}."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from prism.util import FIGURES, RESULTS, write_md_table

# Colour-blind-validated categorical order (fixed, never cycled) and a one-hue sequential ramp.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TYPE_MARKER = {"prose": "o", "maths": "s", "code": "^"}
PAIR_COLOR = {"large": CAT[0], "medium": CAT[1]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100,
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def have(name):
    return (RESULTS / name).exists()


def fig1():
    d = pd.read_csv(RESULTS / "e1_fig1_curves.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i, (a, g) in enumerate(d.groupby("alpha")):
        ax.plot(g.gamma, g.speedup, color=CAT[i], label=f"α = {a:g}")
        k = int(g.gamma_best_int.iloc[0])
        ax.plot([k], [g.speedup[g.gamma == k].iloc[0]], "o", ms=8, color=CAT[i], mec="white", mew=1.5, zorder=5)
        ax.annotate(f"γ*={g.gamma_star_real.iloc[0]:.2f}", (k, g.speedup[g.gamma == k].iloc[0]),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        ax.text(20.3, g.speedup.iloc[-1], f"α = {a:g}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("draft tokens per round, γ")
    ax.set_ylabel("speedup S(γ) (× vs. target alone)")
    ax.set_xlim(0, 22.5)
    ax.set_xticks(range(0, 21, 4))
    save(fig, "fig1_speedup_curves")


def fig2():
    z = np.load(RESULTS / "e1_fig2_plane.npz")
    al, cs, S, G = z["alpha"], z["c"], z["best_speedup"], z["best_gamma"]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.pcolormesh(al, cs, S, cmap=SEQ, shading="nearest", norm=matplotlib.colors.LogNorm(vmin=1, vmax=S.max()))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("best speedup (×, log scale)")
    cs_ = ax.contour(al, cs, G, levels=[0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5], colors=MUTED, linewidths=0.7)
    ax.clabel(cs_, fmt={0.5: "γ≥1", 1.5: "γ≥2", 2.5: "γ≥3", 4.5: "γ≥5", 8.5: "γ≥9", 16.5: "γ≥17", 32.5: "γ≥33"},
              fontsize=6.5, inline=True)
    ax.plot([0, 1], [0, 1], "--", color=INK, lw=1.2)
    ax.text(0.62, 0.66, "α = c", rotation=38, fontsize=8, color=INK)
    if have("e3_beta_stats.csv") and have("timing.csv"):
        st = pd.read_csv(RESULTS / "e3_beta_stats.csv")
        tm = pd.read_csv(RESULTS / "timing.csv").dropna(subset=["c"])
        for _, r in st.iterrows():
            c = float(tm.loc[tm.pair == r.pair, "c"].iloc[0])
            ax.plot(r["mean"], c, TYPE_MARKER[r.type], ms=8, color=PAIR_COLOR[r.pair], mec="white", mew=1.2, zorder=6)
        h = [plt.Line2D([], [], ls="", marker=TYPE_MARKER[t], color=MUTED, ms=7, label=t) for t in TYPE_MARKER]
        h += [plt.Line2D([], [], ls="", marker="o", color=PAIR_COLOR[p], ms=7, label=f"distilgpt2→gpt2-{p}")
              for p in tm.pair.unique()]
        ax.legend(handles=h, loc="upper left", fontsize=6.5, ncol=1)
    ax.set_xlabel("acceptance rate α")
    ax.set_ylabel("cost ratio c (draft step ÷ target step)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)
    save(fig, "fig2_alpha_c_plane")


def fig3():
    d = pd.read_csv(RESULTS / "e2_modelA_main.csv").sort_values("sd_alpha")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(d.sd_alpha, d.gain_pct, "-o", ms=8, color=CAT[0], mec="white", mew=1.2,
            label="adaptive oracle gain over best fixed γ")
    ax.plot(d.sd_alpha, d.landmark_error_pct, "-s", ms=8, color=CAT[1], mec="white", mew=1.2,
            label="landmark formula underestimate")
    for _, r in d.iterrows():
        ax.annotate("κ=∞" if np.isinf(r.kappa) else f"κ={r.kappa:g}", (r.sd_alpha, r.gain_pct), xytext=(-4, 7), textcoords="offset points",
                    fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("SD of per-round acceptance α (m = 0.7, c = 0.05)")
    ax.set_ylabel("percent (%)")
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "fig3_variability_gain")


def fig4():
    if not have("e3_fig4_example.npz"):
        print("skip fig4 (no E3 results)")
        return
    z = np.load(RESULTS / "e3_fig4_example.npz")
    kinds = [k for k in ("prose", "code") if f"{k}_beta" in z]
    fig, axes = plt.subplots(len(kinds), 1, figsize=(6.2, 1.9 * len(kinds)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k, col in zip(axes, kinds, (CAT[0], CAT[2])):
        b, st = z[f"{k}_beta"], z[f"{k}_starts"]
        x = np.arange(len(b))
        ax.plot(x, b, color=col, lw=1.5)
        for s in st:
            ax.axvline(s, color=MUTED, lw=0.5, alpha=0.6, zorder=0)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"β_t ({k})")
        ax.grid(axis="x", visible=False)
        ax.text(1.0, 1.04, f"{len(st)} rounds, c = {float(z[f'{k}_c']):.3f}", transform=ax.transAxes,
                ha="right", fontsize=7, color=MUTED)
    axes[-1].set_xlabel("generated-token position t (after 32-token prefix); grey lines = adaptive round starts")
    save(fig, "fig4_trace_example")


def fig5():
    files = sorted(RESULTS.glob("traces_*_*.npz"))
    if not files:
        print("skip fig5 (no E3 results)")
        return
    types = ["prose", "maths", "code"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, t in zip(axes, types):
        for f in files:
            pair = f.stem.split("_")[1]
            if f.stem.endswith(t):
                b = np.load(f)["beta"].ravel()
                ax.hist(b, bins=bins, density=True, histtype="step", lw=1.8, color=PAIR_COLOR[pair],
                        label=f"→gpt2-{pair}")
        ax.set_xlabel(f"β_t ({t})")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "fig5_trace_hist")


def table1():
    d = pd.read_csv(RESULTS / "e0_checks.csv")
    d.to_csv(RESULTS / "table1_checks.csv", index=False)
    write_md_table(d[["check", "formula_value", "simulated_value", "abs_error", "pass"]], RESULTS / "table1_checks.md")


def table2():
    A = pd.read_csv(RESULTS / "e2_modelA_main.csv")
    rows = []
    for _, r in A.iterrows():
        rows.append(dict(section="A persistent", setting=f"kappa={r.kappa:g}, SD={r.sd_alpha:.3f}",
                         landmark_pred=f"{r.landmark_pred:.3f} ({r.landmark_gamma})",
                         fixed_true_at_landmark_gamma=f"{r.fixed_true_at_landmark_gamma:.3f}",
                         best_fixed=f"{r.best_fixed_true:.3f} ({r.best_fixed_gamma})",
                         adaptive=f"{r.adaptive_oracle:.3f}", gain_pct=f"{r.gain_pct:.1f}",
                         landmark_error_pct=f"{r.landmark_error_pct:.1f}"))
    B = pd.read_csv(RESULTS / "e2_modelB.csv")
    for _, r in B.iterrows():
        rows.append(dict(section="B per-token iid", setting=f"kappa={r.kappa:g}, SD={r.sd_alpha:.3f}",
                         landmark_pred=f"{r.landmark_pred:.3f} ({r.landmark_gamma})",
                         fixed_true_at_landmark_gamma=f"{r.fixed_at_landmark_mean:.3f} ± {r.fixed_at_landmark_sd:.3f}",
                         best_fixed=f"{r.best_fixed_mean:.3f} ± {r.best_fixed_sd:.3f} ({r.best_fixed_gamma})",
                         adaptive=f"history {r.history_adaptive_mean:.3f}; look-ahead {r.lookahead_oracle_mean:.3f}",
                         gain_pct=f"history {r.history_gain_pct:.2f}; look-ahead {r.lookahead_gain_pct:.2f}",
                         landmark_error_pct=f"{r.landmark_error_pct:.2f}"))
    C = pd.read_csv(RESULTS / "e2_modelC.csv")
    for _, r in C.iterrows():
        rows.append(dict(section="C Markov regimes", setting=f"switch s={r.switch_prob:g}",
                         landmark_pred=f"{r.landmark_pred:.3f} ({r.landmark_gamma})",
                         fixed_true_at_landmark_gamma=f"{r.fixed_at_landmark_mean:.3f} ± {r.fixed_at_landmark_sd:.3f}",
                         best_fixed=f"{r.best_fixed_mean:.3f} ± {r.best_fixed_sd:.3f} ({r.best_fixed_gamma})",
                         adaptive=f"state {r.adaptive_state_mean:.3f} ± {r.adaptive_state_sd:.3f}; "
                                  f"per-token {r.adaptive_per_token_mean:.3f} ± {r.adaptive_per_token_sd:.3f}",
                         gain_pct=f"state {r.state_gain_pct:.1f}; per-token {r.per_token_gain_pct:.1f}",
                         landmark_error_pct=f"{r.landmark_error_pct:.1f}"))
    D = pd.read_csv(RESULTS / "e2_modelD.csv")
    for _, r in D.iterrows():
        rows.append(dict(section="D noisy System 1", setting=f"kappa=10, sigma={r.sigma:g}",
                         landmark_pred="", fixed_true_at_landmark_gamma="", best_fixed=f"{r.best_fixed_true:.3f}",
                         adaptive=f"{r.throughput_mean:.3f} ± {r.throughput_sd:.3f}",
                         gain_pct=f"{r.gain_pct:.1f} (survival {r.gain_survival_pct:.0f}% of oracle gain)",
                         landmark_error_pct=""))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "table2_variable_alpha.csv", index=False)
    write_md_table(t, RESULTS / "table2_variable_alpha.md")


def table3():
    if not have("e3_replay.csv"):
        print("skip table3 (no E3 results)")
        return
    d = pd.read_csv(RESULTS / "e3_replay.csv")
    d["cell"] = d.apply(lambda r: f"{r.thr_mean:.3f} ± {r.thr_sd:.3f}", axis=1)
    order = ["no_spec", "fixed_4", "landmark", "best_fixed", "adaptive_oracle", "adaptive_realistic"]
    for label, sub, name in [("measured", d[d.c_label == "measured"], "table3_gpt2"),
                             ("sens", d[d.c_label != "measured"], "table3_gpt2_sensitivity")]:
        p = sub.pivot_table(index=["pair", "type", "c"], columns="policy", values="cell", aggfunc="first")[order]
        g = sub.pivot_table(index=["pair", "type", "c"], columns="policy", values="gamma", aggfunc="first")
        p["landmark_gamma"] = g["landmark"]
        p["best_fixed_gamma"] = g["best_fixed"]
        p["realistic_calibrator"] = g["adaptive_realistic"]
        rec = sub[sub.policy == "adaptive_realistic"].set_index(["pair", "type", "c"])["recovery_pct"]
        p["realistic_recovery_pct"] = rec.round(1)
        p = p.reset_index()
        p["c"] = p["c"].round(4)
        p.to_csv(RESULTS / f"{name}.csv", index=False)
        write_md_table(p, RESULTS / f"{name}.md")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    table1(); table2(); table3()
