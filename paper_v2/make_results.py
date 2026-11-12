"""Tables and figures for the paper from the trace files and analysis outputs. Writes results.json, fig_*.pdf."""
import json, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = "../experiments/qwen/"
CONDS = [("Prose", "sampling", "qwen_prose_traces.npz"), ("Maths", "sampling", "qwen_maths_traces.npz"), ("Code", "sampling", "qwen_code_traces.npz"),
         ("Prose", "greedy", "qwen_prose_greedy_traces.npz"), ("Maths", "greedy", "qwen_maths_greedy_traces.npz"),
         ("Code", "greedy", "qwen_code_greedy_traces.npz"),
         ("Multi-turn chat", "specbench", "specbench:mt_bench"), ("Translation", "specbench", "specbench:translation"),
         ("Summarisation", "specbench", "specbench:summarization"), ("Question answering", "specbench", "specbench:qa"),
         ("Maths reasoning", "specbench", "specbench:math_reasoning"), ("Retrieval (RAG)", "specbench", "specbench:rag")]
GAMMAS = [1, 2, 3, 4, 5, 6, 8, 10]
rng = np.random.default_rng(0)
R, B = 20, 2000


def load(f):
    """Per-position arrays for one condition; Spec-Bench task types are slices of one trace file."""
    if f.startswith("specbench:"):
        z = np.load(A + "specbench_traces.npz")
        ix = z["categories"] == f.split(":")[1]
        return {k: z[k][ix] for k in ("own_beta", "own_a")}
    return np.load(A + f)


def per_passage(a, g):
    """tokens and rounds per passage for fixed gamma (complete rounds), summed over R exact replays."""
    n, T = a.shape
    out = np.zeros((n, 2))
    reps = 1 if set(np.unique(a)) <= {0.0, 1.0} else R
    for _ in range(reps):
        acc = rng.random((n, T)) < a
        for i in range(n):
            t = 0
            while t + g < T:
                k = 0
                while k < g and acc[i, t + k]:
                    k += 1
                out[i, 0] += k + 1; out[i, 1] += 1; t += k + 1
    return out


res = {"formula": []}
for name, mode, f in CONDS:
    z = load(f)
    beta, a = z["own_beta"].astype(float), z["own_a"].astype(float)
    n, T = beta.shape
    m = beta.mean()
    acf = lambda k: float(((beta[:, :-k] - m) * (beta[:, k:] - m)).sum() / ((beta - m) ** 2).sum())
    pm = beta.mean(1)
    row = dict(name=name, mode=mode, n=n, alpha=float(m), sd=float(beta.std()), acf1=acf(1), acf5=acf(5),
               sd_passage=float(pm.std()), err={})
    ix = rng.integers(0, n, (B, n))
    bsum, bcnt = beta.sum(1), np.full(n, T)
    for g in GAMMAS:
        pp = per_passage(a, g)
        exact = pp[:, 0].sum() / pp[:, 1].sum()
        lm = (1 - m ** (g + 1)) / (1 - m)
        al_b = bsum[ix].sum(1) / bcnt[ix].sum(1)
        ex_b = pp[ix, 0].sum(1) / pp[ix, 1].sum(1)
        e_b = 100 * (((1 - al_b ** (g + 1)) / (1 - al_b)) / ex_b - 1)
        row["err"][g] = dict(landmark=float(lm), exact=float(exact), err=float(100 * (lm / exact - 1)),
                             lo=float(np.percentile(e_b, 2.5)), hi=float(np.percentile(e_b, 97.5)))
    res["formula"].append(row)

# ---- adaptive results parsed from the analysis outputs
def parse(path):
    out, cur, cost = {}, None, None
    for line in open(path):
        m1 = re.match(r"===== (\S+):", line)
        if m1: cur = m1.group(1); out[cur] = {}; continue
        m2 = re.match(r"\s+c=([\d.]+), v=([\d.]+): best fixed gamma per fold \[(\d+), (\d+)\]", line)
        if m2: cost = f"{m2.group(1)}/{m2.group(2)}"; out[cur][cost] = {"gamma": [int(m2.group(3)), int(m2.group(4))]}; continue
        m3 = re.match(r"\s+(\w+)\s+speedup ([\d.]+)x\s+gain over best fixed\s+([+-][\d.]+)% \[([+-][\d.]+),([+-][\d.]+)\]", line)
        if m3: out[cur][cost][m3.group(1)] = dict(speedup=float(m3.group(2)), gain=float(m3.group(3)), lo=float(m3.group(4)), hi=float(m3.group(5))); continue
        m4 = re.match(r"\s+share of oracle gain recovered.*: (-?\d+)%", line)
        if m4: out[cur][cost]["recovered"] = int(m4.group(1))
        m5 = re.match(r"\s+exact replay, first 30 passages, gamma=(\d): ([\d.]+) \[([\d.]+),([\d.]+)\]", line)
        if m5: out[cur].setdefault("first30", {})[m5.group(1)] = [float(m5.group(2)), float(m5.group(3)), float(m5.group(4))]
    return out
ad = {}
for p in ("results.txt", "results_greedy.txt", "results_specbench.txt"):
    ad.update(parse(A + p))
res["adaptive"] = ad
res["timing"] = json.load(open(A + "timing.json"))["runs"]
res["sim"] = json.load(open("sim_numbers.json"))
# share of the error explained by between-passage differences alone (harmonic mean over passage-level alpha)
gfun = lambda al, G: (1 - al ** (G + 1)) / (1 - al)
for (name, mode, f), row in zip(CONDS, res["formula"]):
    ap = np.clip(load(f)["own_beta"].astype(float).mean(1), 1e-6, 1 - 1e-6)
    row["passage_harmonic_err"] = {g: float(100 * (row["err"][g]["landmark"] / (len(ap) / np.sum(1 / gfun(ap, g))) - 1)) for g in GAMMAS}
json.dump(res, open("results.json", "w"), indent=1)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False})
COL = ["#2a78d6", "#eb6834", "#1baf7a"]; MK = ["o", "s", "^"]

# Figure 1: the sign flip in the two synthetic models
sys.path.insert(0, A)
import prop3_markov as P
from scipy import stats
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
m, c = 0.7, 0.05
sds, errs = [], []
for kappa in np.geomspace(4, 2000, 60):
    a_, b_ = m * kappa, (1 - m) * kappa
    mom = lambda k: float(np.prod([(a_ + j) / (a_ + b_ + j) for j in range(k)]))
    thr = [sum(mom(k) for k in range(G + 1)) / (G * c + 1) for G in range(41)]
    G = 6; lm = (1 - m ** (G + 1)) / (1 - m) / (G * c + 1)        # gamma = 6 is the best fixed length for constant alpha = 0.7
    sds.append(np.sqrt(m * (1 - m) / (kappa + 1))); errs.append(100 * (lm / thr[G] - 1))
ax[0].axhline(0, color="#52514e", lw=0.8)
ax[0].plot(sds, errs, color=COL[0], lw=2)
ax[0].set_xlabel(r"SD of per-round acceptance $\alpha$"); ax[0].set_ylabel("error of landmark formula (%)")
ax[0].set_title("(a) Round-level persistence: formula too low", fontsize=8, loc="left")
ss = np.geomspace(0.005, 0.5, 40)
for j, (cc, ls) in enumerate(((0.05, "-"), (0.2, "--"))):
    e = []
    for s in ss:
        fx = max((P.throughput_obs(g, g, s, cc), g) for g in range(1, 21))
        G = fx[1]; e.append(100 * (((1 - 0.675 ** (G + 1)) / (1 - 0.675)) / (G * cc + 1) / fx[0] - 1))
    ax[1].plot(ss, e, color=COL[j], lw=2, ls=ls, label=f"$c={cc}$")
ax[1].axhline(0, color="#52514e", lw=0.8); ax[1].set_xscale("log")
ax[1].set_xlabel("regime switch probability per token, $s$"); ax[1].set_ylim(ax[0].get_ylim()[0] * 0 - 1, 12)
ax[1].set_title("(b) Token-level persistence: formula too high", fontsize=8, loc="left"); ax[1].legend(loc="upper right")
fig.tight_layout(); fig.savefig("fig_signflip.pdf"); plt.close(fig)

# Figure 2: formula error against gamma on real traces
fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=True)
for p, (mode, title) in enumerate((("sampling", "(a) Sampling, $T=1$"), ("greedy", "(b) Greedy"), ("specbench", "(c) Spec-Bench, sampling"))):
    rows = [r for r in res["formula"] if r["mode"] == mode]
    ax[p].axhline(0, color="#52514e", lw=0.8)
    if mode != "specbench":
        for j, r in enumerate(rows):
            e = [r["err"][g] for g in GAMMAS]
            ax[p].fill_between(GAMMAS, [x["lo"] for x in e], [x["hi"] for x in e], color=COL[j], alpha=0.15, lw=0)
            ax[p].plot(GAMMAS, [x["err"] for x in e], color=COL[j], lw=1.6, marker=MK[j], ms=4, mec="white", mew=0.6, label=r["name"])
        ax[p].legend(loc="upper left")
    else:
        for r in rows:
            ax[p].plot(GAMMAS, [r["err"][g]["err"] for g in GAMMAS], color="#8a8983", lw=1.1, marker="o", ms=2.5)
        top = max(rows, key=lambda r: r["err"][6]["err"]); bot = min(rows, key=lambda r: r["err"][10]["err"])
        ax[p].annotate(top["name"], (6, top["err"][6]["err"]), xytext=(4.3, 5.6), fontsize=7, color="#52514e",
                       arrowprops=dict(arrowstyle="-", color="#8a8983", lw=0.6))
        ax[p].annotate(bot["name"], (8, bot["err"][8]["err"]), xytext=(2.6, -1.55), fontsize=7, color="#52514e",
                       arrowprops=dict(arrowstyle="-", color="#8a8983", lw=0.6))
    ax[p].set_title(title, fontsize=8, loc="left"); ax[p].set_xlabel(r"draft tokens per round, $\gamma$")
ax[0].set_ylabel("error of landmark formula (%)")
fig.tight_layout(); fig.savefig("fig_error.pdf"); plt.close(fig)

# digest
for r in res["formula"]:
    e = r["err"]
    print(f"{r['mode']:<9} {r['name']:<19} n={r['n']:<3} a={r['alpha']:.3f} sd={r['sd']:.2f} acf={r['acf1']:.2f}/{r['acf5']:.2f} sdP={r['sd_passage']:.2f} | "
          + "  ".join(f"g{g}:{e[g]['err']:+.1f}[{e[g]['lo']:+.1f},{e[g]['hi']:+.1f}]" for g in (3, 6, 10)))
