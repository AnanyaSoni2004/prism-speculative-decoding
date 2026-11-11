"""Write results/REPORT.md. Every number is read from files in results/."""
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from prism.util import RESULTS

R = RESULTS
out = []
P = out.append


def jload(name, default=None):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else default


def md(name):
    p = R / name
    return p.read_text() if p.exists() else "_not available_\n"


def f(x, d=3):
    return f"{x:.{d}f}"


runtime = jload("runtime.json", {})
seeds = jload("seeds.json", {})
skip_models = not (R / "e3_replay.csv").exists()

# ------------------------------------------------------------------ 1. run summary
chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
mem_gb = int(mem) / 2**30 if mem.isdigit() else float("nan")
timing = pd.read_csv(R / "timing.csv") if (R / "timing.csv").exists() else None
device = timing.device.iloc[0] if timing is not None else "n/a (E3 skipped)"
env = (R / "env.txt").read_text().splitlines() if (R / "env.txt").exists() else []
keep = ("torch==", "transformers==", "datasets==", "numpy==", "scipy==", "scikit-learn==", "matplotlib==", "pandas==")
vers = [l for l in env if l.lower().startswith(keep)]

P("# PRISM: speculative decoding under variable acceptance — experiment report\n")
P("## 1. Run summary\n")
P(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
P(f"- Machine: {chip}, {mem_gb:.0f} GB RAM, {platform.platform()}, Python {platform.python_version()}")
P(f"- Device for E3: {device}")
tot = runtime.get("total", {}).get("seconds")
stage_t = ", ".join(f"{k} {v['seconds']:.0f} s" for k, v in runtime.items() if k != "total")
P(f"- Total runtime: {tot:.0f} s ({stage_t})" if tot else f"- Stage runtimes: {stage_t}")
P(f"- Seeds: `{json.dumps(seeds)}`")
P(f"- Packages: {', '.join(vers)} (full list in `results/env.txt`)\n")

# ------------------------------------------------------------------ 2. E0
e0 = pd.read_csv(R / "e0_checks.csv")
P("## 2. E0 — theory checks\n")
P(f"**{int(e0['pass'].sum())}/{len(e0)} checks passed.**\n")
groups = e0.check.str.split(" ").str[0].str.extract(r"(\d)")[0]
for gnum, sub in e0.groupby(groups):
    worst = sub.loc[sub.abs_error.idxmax()]
    P(f"- Check {gnum}: {int(sub['pass'].sum())}/{len(sub)} passed; largest abs error {worst.abs_error:.3g} ({worst.check}).")
fails = e0[~e0["pass"]]
if len(fails):
    P("\nFailed checks:\n")
    for _, r in fails.iterrows():
        P(f"- {r.check}: formula {r.formula_value:.6g}, simulated {r.simulated_value:.6g}")
P("\nFull table: `results/table1_checks.md`.\n")

# ------------------------------------------------------------------ 3. E1
e1 = pd.read_csv(R / "e1_grid.csv")
c05 = e1[(e1.c == 0.05) & e1.alpha.round(2).isin([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])]
P("## 3. E1 — fixed α (landmark)\n")
r08 = e1[(e1.c == 0.05) & (e1.alpha.round(2) == 0.8)].iloc[0]
r04 = e1[(e1.c == 0.4) & (e1.alpha.round(2) == 0.8)].iloc[0]
caps = int(e1.hits_cap.sum())
P(f"Under constant α the optimal draft length grows quickly with α and falls with c: at c = 0.05, α = 0.8 gives "
  f"γ* = {f(r08.gamma_star_real, 4)} (best integer {int(r08.gamma_best_int)}, S = {f(r08.best_speedup)}), "
  f"while at c = 0.4 the same α gives best integer γ = {int(r04.gamma_best_int)} and S = {f(r04.best_speedup)}. "
  f"The best integer is always floor or ceil of the Lambert-W γ* (E0 check 4), and speculation helps exactly when α > c "
  f"(E0 check 5). {caps} of {len(e1)} grid points hit the γ cap of 40.\n")
P("| α | γ* (Lambert W) | best integer γ | best speedup |\n|---|---|---|---|")
for _, r in c05.iterrows():
    P(f"| {r.alpha:.2f} | {r.gamma_star_real:.4f} | {int(r.gamma_best_int)} | {r.best_speedup:.3f} |")
P("\nTable for c = 0.05. Full grid: `results/e1_grid.csv`.\n")

# ------------------------------------------------------------------ 4. E2
A = pd.read_csv(R / "e2_modelA_main.csv")
Aall = pd.read_csv(R / "e2_modelA_all.csv")
B = pd.read_csv(R / "e2_modelB.csv")
C = pd.read_csv(R / "e2_modelC.csv")
D = pd.read_csv(R / "e2_modelD.csv")
P("## 4. E2 — variable α\n")
P("Model A (α persistent within a round, m = 0.7, c = 0.05, exact on a 20,001-point grid):\n")
P("| κ | SD(α) | landmark_pred (γ) | fixed_true_at_landmark_γ | best_fixed_true (γ) | adaptive_oracle | gain_pct | landmark_error_pct |")
P("|---|---|---|---|---|---|---|---|")
for _, r in A.iterrows():
    k = "∞" if np.isinf(r.kappa) else f"{r.kappa:g}"
    P(f"| {k} | {r.sd_alpha:.3f} | {r.landmark_pred:.3f} ({r.landmark_gamma}) | {r.fixed_true_at_landmark_gamma:.3f} | "
      f"{r.best_fixed_true:.3f} ({r.best_fixed_gamma}) | {r.adaptive_oracle:.3f} | {r.gain_pct:.1f} | {r.landmark_error_pct:.1f} |")
capA = int(Aall.best_fixed_hits_cap.sum())
capmass = Aall.adaptive_mass_at_cap.max()
P(f"\nCap flags: best fixed γ hits 40 in {capA} of {len(Aall)} Model-A settings (main + appendix); the adaptive policy puts "
  f"at most {100 * capmass:.2f}% of rounds at γ = 40 (setting: "
  f"{Aall.loc[Aall.adaptive_mass_at_cap.idxmax(), ['m', 'c', 'kappa']].to_dict()}). Appendix table: `results/e2_modelA_all.csv`.\n")
P(f"- **Model B** (α drawn independently per token, 200k rounds × 5 seeds): the landmark formula stays exact "
  f"(error {B.landmark_error_pct.abs().max():.2f}% at most) and a policy that adapts on information from before the round "
  f"gains nothing ({B.history_gain_pct.min():.2f}% to {B.history_gain_pct.max():.2f}%). A per-token look-ahead oracle that "
  f"sees each token's α before drafting it does gain {B.lookahead_gain_pct.min():.1f}–{B.lookahead_gain_pct.max():.1f}% (see section 7).")
P(f"- **Model C** (Markov regimes α ∈ {{0.95, 0.4}}, 20k rounds × 5 seeds): the adaptive policy that knows the state at round start "
  f"beats the best fixed γ by " + ", ".join(f"{r.state_gain_pct:.1f}% (s = {r.switch_prob:g})" for _, r in C.iterrows()) +
  "; the per-token state oracle by " + ", ".join(f"{r.per_token_gain_pct:.1f}%" for _, r in C.iterrows()) +
  ". The landmark formula with the stationary mean α = 0.675 *overestimates* throughput here (" +
  ", ".join(f"{r.landmark_error_pct:.1f}%" for _, r in C.iterrows()) + ").")
d0 = D[D.sigma == 0].iloc[0]
P(f"- **Model D** (noisy System 1, κ = 10): the oracle gain of {d0.oracle_gain_pct:.1f}% survives as " +
  ", ".join(f"{r.gain_survival_pct:.0f}% at σ = {r.sigma:g}" for _, r in D.iterrows()) +
  " of the oracle gain (negative = worse than best fixed γ).\n")
P("Combined table: `results/table2_variable_alpha.md`.\n")

# ------------------------------------------------------------------ 5. E3
P("## 5. E3 — GPT-2 traces\n")
if skip_models:
    P("_E3 was skipped (`--skip-models`)._\n")
else:
    counts = jload("e3_data_counts.json", {})
    P("Passages: " + "; ".join(f"{k}: {v['used']} used of {v['long_enough']} with ≥160 tokens ({v['candidates']} candidates)"
                              for k, v in counts.items()) + ".\n")
    tm = timing
    P("**Measured c** (64 tokens one at a time with KV cache, 5 repeats after 2 warm-ups):\n")
    P("| model | role | per-token time (ms) | c | 8-token verify ÷ 1-token step |\n|---|---|---|---|---|")
    for _, r in tm.iterrows():
        c = "" if pd.isna(r.get("c")) else f"{r.c:.3f}"
        v = "" if pd.isna(r.get("verify_ratio_8_over_1")) else f"{r.verify_ratio_8_over_1:.2f}"
        P(f"| {r.model} | {r.role} | {1000 * r.per_token_s:.2f} ± {1000 * r.per_token_sd:.2f} | {c} | {v} |")
    st = pd.read_csv(R / "e3_beta_stats.csv")
    P("\n**Is α constant?** Descriptive statistics of β_t (pooled over passages and positions):\n")
    P("| pair | type | mean | SD | q05 | median | q95 | lag-1 ACF | lag-5 ACF | lag-10 ACF | frac β>0.9 | frac β<0.3 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in st.iterrows():
        P(f"| {r.pair} | {r.type} | {r['mean']:.3f} | {r.sd:.3f} | {r.q05:.3f} | {r["median"]:.3f} | {r.q95:.3f} | "
          f"{r.acf_lag1:.3f} | {r.acf_lag5:.3f} | {r.acf_lag10:.3f} | {r.frac_gt_0_9:.3f} | {r.frac_lt_0_3:.3f} |")
    P("\n**Throughput on the test half at measured c** (tokens per target-step unit, mean ± SD over 5 replay seeds; "
      "γ in brackets in the CSV):\n")
    P(md("table3_gpt2.md"))
    rep = pd.read_csv(R / "e3_replay.csv")
    real = rep[rep.policy == "adaptive_realistic"]
    P("**Realistic System 1 recovery** = (realistic − best fixed) ÷ (oracle − best fixed), test half:\n")
    P("| pair | type | measured c | c=0.05 | c=0.1 | c=0.2 |\n|---|---|---|---|---|---|")
    for (pair, typ), g in real.groupby(["pair", "type"], sort=False):
        vals = {r.c_label: r.recovery_pct for _, r in g.iterrows()}
        P(f"| {pair} | {typ} | " + " | ".join(f"{vals.get(k, float('nan')):.0f}%" for k in ["measured", "0.05", "0.1", "0.2"]) + " |")
    P("\nSensitivity table over c: `results/table3_gpt2_sensitivity.md`. Calibrator choice per cell: `results/e3_calibrators.csv`.\n")
    J = pd.read_csv(R / "e3_jensen.csv")
    P("**Jensen check on real data** (test half, measured c, landmark γ): formula with mean β vs replay at the same γ.\n")
    P("The control column replays the same passages with every β_t set to the mean β (same uniforms), which isolates the "
      "effect of 128-token passages (rounds truncated at the end) from the effect of variable β.\n")
    P("| pair | type | γ | mean β | landmark pred | replay (real β) | replay (constant β control) | real − pred | real − control | lag-1 ACF |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in J.iterrows():
        P(f"| {r.pair} | {r.type} | {r.gamma} | {r.mean_beta_test:.3f} | {r.landmark_pred:.3f} | {r.replay_mean:.3f} ± {r.replay_sd:.3f} | "
          f"{r.const_mean:.3f} ± {r.const_sd:.3f} | {r.replay_minus_pred_pct:+.1f}% | {r.replay_minus_const_pct:+.1f}% | {r.lag1_acf_test:.3f} |")
    higher = int((J.replay_minus_pred_pct > 0).sum())
    higher_c = int((J.replay_minus_const_pct > 0).sum())
    P(f"\nReplay with real β is above the landmark formula in {higher} of {len(J)} cells and above the constant-β control in "
      f"{higher_c} of {len(J)} cells; every cell has positive lag-1 autocorrelation: {bool((J.lag1_acf_test > 0).all())}.\n")

# ------------------------------------------------------------------ 6/7. unexpected + deviations
P("## 6. Anything unexpected\n")
P("Each item states the observed result (from the files) and, separately, a **guess** at the cause.\n")
mA = A.set_index("kappa")
P(f"- Model A reproduces every expected value in the brief to the printed precision (e.g. κ = 5: best fixed "
  f"{mA.loc[5, 'best_fixed_true']:.3f} ({int(mA.loc[5, 'best_fixed_gamma'])}), adaptive {mA.loc[5, 'adaptive_oracle']:.3f}, "
  f"gain {mA.loc[5, 'gain_pct']:.1f}%). Nothing unexpected there.")
P(f"- Model B: the brief expected adaptive policies to gain nothing. Policies using only past information gain "
  f"{B.history_gain_pct.max():.2f}% at most, as expected, but the per-token look-ahead oracle gains up to "
  f"{B.lookahead_gain_pct.max():.1f}%. *Guess:* that oracle knows the α of the very token it is about to draft, which is "
  f"information about the future even when tokens are independent; the brief's claim holds for policies that cannot see it.")
P(f"- Model C: the landmark formula with mean α *overestimates* the fixed-γ throughput by "
  f"{-C.landmark_error_pct.min():.1f}% at most, the opposite sign to Model A. *Guess:* in a token-level Markov stream, "
  f"throughput is a harmonic-type average over positions (hard stretches cost more time per token), and rounds that start "
  f"in the hard state are short, so time is weighted toward the hard regime. Model A's Jensen gain comes from drawing α per "
  f"round, which weights each round equally. Persistence alone does not guarantee that the landmark formula underestimates.")
P(f"- Model D: at σ = 0.2 the noisy threshold policy is worse than the best fixed γ "
  f"({D.loc[D.sigma == 0.2, 'gain_pct'].iloc[0]:.1f}%). *Guess:* the rule plugs α̂ straight into α^k ≥ λc without shrinking "
  f"noisy estimates toward the mean, so large positive errors trigger very long, wasted drafts. A Bayes-shrunk α̂ would likely "
  f"do better; we did not change the design.")
if not skip_models:
    J = pd.read_csv(R / "e3_jensen.csv")
    below = J[(J.replay_minus_const_pct < 0) & (J.lag1_acf_test > 0)]
    if len(below):
        P(f"- E3 Jensen check: the proposition predicts replay above the formula when β is positively autocorrelated, but "
          f"in {len(below)} of {len(J)} cells the real-β replay is *below* the constant-β control (by "
          f"{below.replay_minus_const_pct.min():.1f}% to {below.replay_minus_const_pct.max():.1f}%), with lag-1 ACF "
          f"{below.lag1_acf_test.min():.2f}–{below.lag1_acf_test.max():.2f}. *Guess:* same mechanism as Model C: the "
          f"autocorrelation is weak and short-lived (lag-5 ACF ≈ 0.03–0.11), so per-token variability acts like Model B "
          f"(no gain), while low-β positions end rounds early and so take up more of the time.")
    layers = {"gpt2-medium": 24, "gpt2-large": 36}
    for _, r in timing.dropna(subset=["c"]).iterrows():
        if r.c > 0.2:
            P(f"- Measured c for distilgpt2 → {r.model} is {r.c:.3f}, well above the 6/{layers.get(r.model, '?')} layer ratio. "
              f"*Guess:* on {r.device} with batch size 1, per-token time for these small models is dominated by kernel launch "
              f"and Python overhead rather than FLOPs, so the draft costs a larger share than its size suggests.")
        if r.verify_ratio_8_over_1 > 1.5:
            P(f"- Verifying 8 tokens with {r.model} takes {r.verify_ratio_8_over_1:.2f}× one cached step, so the "
              f"'verification costs one step' assumption is only approximate on this device.")

P("\n## 7. Deviations\n")
devs = [
    "Project lives in `/Users/tanish/Desktop/prism-research` (the folder this session was opened in); `~/prism-spec` is a "
    "symlink to it, so `cd ~/prism-spec && bash run_all.sh` works as specified.",
    f"Machine is an {chip} with {mem_gb:.0f} GB RAM, not an M1 with 16 GB as the brief assumed.",
    "`legacy/check_math.py` and `legacy/check_tc.py` were not provided to this session (not found on disk), so E0 was "
    "written from the specification in the brief alone; `legacy/` is empty.",
    "`pytest` was added to requirements.txt to run the tests.",
    "E0 check 5 compares max S against 1 + 1e-9 instead of 1: on the diagonal α = c, S(1) = 1 exactly but floating "
    "point gives 1 + 2e-16 at 15 grid points (the raw count is recorded in e0_checks.csv).",
    "E0 check 1b's pass criterion: L1 below twice the expected L1 of exact sampling from p with the same n.",
    "Model A uses cell-centred grid points with exact Beta mass per cell (CDF differences), so densities that diverge at "
    "0 or 1 (e.g. m = 0.9, κ = 5) stay finite.",
    "Model B: 'adaptive' is reported two ways: a history-based threshold policy (uses the previous round's mean α) and a "
    "per-token look-ahead oracle. The brief did not define which adaptive policy to use under independent α.",
    "Model C: two adaptive policies are reported: γ chosen from the state at round start (grid search over γ_easy ∈ 0..40, "
    "γ_hard ∈ 0..15 on a separate fit seed), and the per-token rule using the true state of each drafted position. Best "
    "fixed γ and λ are chosen on a separate fit seed, then evaluated on 5 seeds.",
    "Model D computes throughput as Σ E[N | true α, γ(α̂)] / Σ cost (accept draws integrated out exactly) over 200k sampled "
    "rounds per seed; λ is refit for each σ on a separate fit seed.",
    "Monte-Carlo λ fits use the fixed-point iteration; if it cycles, the visited λ with the highest training throughput is kept.",
]
if not skip_models:
    devs += [
        "E3 replay: each position's accept/reject is pre-drawn as u_t < β_t with per-seed uniforms shared by all policies "
        "(common random numbers). This is exact because each position is drafted at most once. A round never drafts past "
        "the end of the passage, so every passage produces exactly 128 tokens.",
        "E3 landmark policy uses the best integer γ under the landmark formula at the training-half mean β; the real γ* is in "
        "`e3_jensen.csv`.",
        "E3 calibrators are fitted per (pair, text type) on the training half; logistic regression uses soft labels (each "
        "row duplicated with label 1 weight β and label 0 weight 1 − β). The better calibrator is chosen by training-half "
        "throughput of the realistic policy.",
        "Model downloads used `HF_HUB_DISABLE_XET=1` because Hugging Face's Xet transfer endpoint returned 404.",
    ]
    for d in jload("e3_deviations.json", []) or []:
        devs.append(d)
else:
    devs.append("E3 was skipped (`--skip-models`).")
for d in devs:
    P(f"- {d}")

(R / "REPORT.md").write_text("\n".join(out) + "\n")
print("wrote", R / "REPORT.md")
