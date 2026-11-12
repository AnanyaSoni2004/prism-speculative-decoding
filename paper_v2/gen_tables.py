"""Writes the LaTeX table bodies from results.json so no number is typed by hand."""
import json
import numpy as np
r = json.load(open("results.json"))
def S(x):
    t = f"{x:+.1f}"
    if t in ("-0.0", "+0.0"): return "0.0"
    return t.replace("-", "$-$")

# Table: formula error on real traces
L = []
last = None
head = {"sampling": "Sampling ($T=1$), base models", "greedy": "Greedy, base models", "specbench": "Spec-Bench, sampling, instruct models"}
for row in r["formula"]:
    if row["mode"] != last:
        if last: L.append(r"\addlinespace")
        L.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{head[row['mode']]}}}}}\\")
        last = row["mode"]
    e = row["err"]
    cell = lambda g: f"{S(e[g]['err'])} [{S(e[g]['lo'])}, {S(e[g]['hi'])}]"
    L.append(f"{row['name']} & {row['n']} & {row['alpha']:.3f} & {row['sd']:.2f} & {row['acf1']:.2f} / {row['acf5']:.2f} & "
             f"{cell('3')} & {cell('6')} & {cell('10')}\\\\")
open("tab_formula.tex", "w").write("\n".join(L) + "\n")

# Table: adaptive rules
key = {"Prose/sampling": "prose", "Maths/sampling": "maths", "Code/sampling": "code", "Prose/greedy": "prose_greedy",
       "Maths/greedy": "maths_greedy", "Code/greedy": "code_greedy", "Multi-turn chat/specbench": "mt_bench",
       "Translation/specbench": "translation", "Summarisation/specbench": "summarization",
       "Question answering/specbench": "qa", "Maths reasoning/specbench": "math_reasoning", "Retrieval (RAG)/specbench": "rag"}
L, last = [], None
for row in r["formula"]:
    a = r["adaptive"][key[f"{row['name']}/{row['mode']}"]]
    if row["mode"] != last:
        if last: L.append(r"\addlinespace")
        L.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{head[row['mode']]}}}}}\\")
        last = row["mode"]
    x, y = a["0.2/1.0"], a["0.66/1.2"]
    g = lambda d, k: S(d[k]["gain"]) if k in d else "--"
    sp = lambda d, k: f"{d[k]['speedup']:.2f}" if k in d else "--"
    L.append(f"{row['name']} & {sp(x,'fixed')} & {g(x,'heuristic')} & {g(x,'pertoken')} & {g(x,'survival')} & {g(x,'oracle')} & "
             f"{sp(y,'fixed')} & {sp(y,'pertoken')} & {sp(y,'survival')} & {sp(y,'oracle')}\\\\")
open("tab_adaptive.tex", "w").write("\n".join(L) + "\n")

# Table: Model C, threshold rule against the optimum
L = []
for row in r["sim"]["C"]:
    L.append(f"{row['c']} & {row['s']} & {row['fixed']:.3f} ({row['best_gamma']}) & {S(row['landmark_error_pct'])} & {row['hard_start_pct']:.0f} & "
             f"{row['threshold']:.3f} ({row['thr_g'][0]}, {row['thr_g'][1]}) & {row['optimum']:.3f} ({row['opt_g'][0]}, {row['opt_g'][1]}) & {row['thr_loss_pct']:.1f}\\\\")
    if row["s"] == 0.5 and row["c"] == 0.05: L.append(r"\addlinespace")
open("tab_modelc.tex", "w").write("\n".join(L) + "\n")

# numbers quoted in the text
g = lambda a, G: (1 - a ** (G + 1)) / (1 - a)
c, v = 0.66, 1.2
lo, hi = 0.5, 0.99
for _ in range(60):
    mid = (lo + hi) / 2
    if max(g(mid, G) / (G * c + v) for G in range(1, 40)) > 1: hi = mid
    else: lo = mid
print("alpha above which some gamma>=1 beats no speculation at c=0.66, v=1.2:", round(hi, 3), " gamma=1 only:", c + v - 1)
t = r["timing"]
vv = [x for run in t for m, x in run["v"].items() if m != "1"]
print("v(m), m>=2: median", round(float(np.median(vv)), 3), "min", round(min(vv), 2), "max", round(max(vv), 2))
print("per-layer ms:", round(t[0]["draft_ms"][0] / 24, 2), round(t[0]["target_ms"][0] / 36, 2))
for row in r["sim"]["A"]: print("A", row["kappa"], round(row["sd"], 3), "g6 err", round(row["err_at_gamma6_pct"], 1), "2nd", round(row["second_order_at_gamma6_pct"], 1), "best g", row["best_gamma"], "oracle gain", round(row["oracle_gain_pct"], 1))
ph = [(row["name"], row["mode"], round(row["passage_harmonic_err"]["6"] / row["err"]["6"]["err"], 2)) for row in r["formula"][:6]]
print("share of gamma=6 error explained by passage-level harmonic:", ph)
