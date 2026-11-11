# PRISM: speculative decoding under variable acceptance — experiment report

## 1. Run summary

- Date: 2026-11-11 15:47
- Machine: Apple M3, 8 GB RAM, macOS-27.2-arm64-arm-64bit, Python 3.12.11
- Device for E3: mps
- Total runtime: 226 s (e0 1 s, e1 0 s, e2 39 s, e3 178 s)
- Seeds: `{"e0": 20261110, "e1": "none (deterministic)", "e2": {"fit": 7001, "eval": [101, 202, 303, 404, 505]}, "e3": {"split": 42, "data": 1234, "replay": [11, 22, 33, 44, 55]}}`
- Packages: datasets==5.2.0, matplotlib==3.11.2, numpy==2.5.4, pandas==3.1.1, scikit-learn==1.9.1, scipy==1.18.1, torch==2.15.0, transformers==5.22.2 (full list in `results/env.txt`)

## 2. E0 — theory checks

**38/38 checks passed.**

- Check 1: 2/2 passed; largest abs error 0.000229 (1b sampled draws (200k): L1(empirical, p); formula = expected L1 of exact sampling).
- Check 2: 1/1 passed; largest abs error 5.55e-16 (2 beta = 1 - 0.5||p-q||_1: max error over 1000 pairs).
- Check 3: 16/16 passed; largest abs error 0.0113 (3 E[N] alpha=0.9 gamma=10 (200k rounds)).
- Check 4: 14/14 passed; largest abs error 7.95e-05 (4 known value: S at best integer (0.8, 0.05) = 3.092).
- Check 5: 1/1 passed; largest abs error 0 (5 max_{gamma>=1} S > 1 iff alpha > c: disagreeing points on 99x99 grid (S>1+1e-9)).
- Check 6: 4/4 passed; largest abs error 0.431 (6ii exact E[N] >= landmark at alpha=m (formula=landmark 3.0588)).

Full table: `results/table1_checks.md`.

## 3. E1 — fixed α (landmark)

Under constant α the optimal draft length grows quickly with α and falls with c: at c = 0.05, α = 0.8 gives γ* = 7.8567 (best integer 8, S = 3.092), while at c = 0.4 the same α gives best integer γ = 2 and S = 1.356. The best integer is always floor or ceil of the Lambert-W γ* (E0 check 4), and speculation helps exactly when α > c (E0 check 5). 0 of 455 grid points hit the γ cap of 40.

| α | γ* (Lambert W) | best integer γ | best speedup |
|---|---|---|---|
| 0.50 | 3.0877 | 3 | 1.630 |
| 0.60 | 4.0648 | 4 | 1.921 |
| 0.70 | 5.4803 | 6 | 2.353 |
| 0.80 | 7.8567 | 8 | 3.092 |
| 0.90 | 13.2916 | 13 | 4.674 |
| 0.95 | 21.1129 | 21 | 6.600 |

Table for c = 0.05. Full grid: `results/e1_grid.csv`.

## 4. E2 — variable α

Model A (α persistent within a round, m = 0.7, c = 0.05, exact on a 20,001-point grid):

| κ | SD(α) | landmark_pred (γ) | fixed_true_at_landmark_γ | best_fixed_true (γ) | adaptive_oracle | gain_pct | landmark_error_pct |
|---|---|---|---|---|---|---|---|
| ∞ | 0.000 | 2.353 (6) | 2.353 | 2.353 (6) | 2.353 | 0.0 | 0.0 |
| 50 | 0.064 | 2.353 (6) | 2.394 | 2.394 (6) | 2.437 | 1.8 | 1.7 |
| 20 | 0.100 | 2.353 (6) | 2.452 | 2.452 (6) | 2.579 | 5.2 | 4.2 |
| 10 | 0.138 | 2.353 (6) | 2.538 | 2.556 (7) | 2.877 | 12.6 | 7.9 |
| 5 | 0.187 | 2.353 (6) | 2.684 | 2.783 (10) | 3.619 | 30.0 | 14.1 |

Cap flags: best fixed γ hits 40 in 0 of 60 Model-A settings (main + appendix); the adaptive policy puts at most 32.85% of rounds at γ = 40 (setting: {'m': 0.9, 'c': 0.05, 'kappa': 5.0}). Appendix table: `results/e2_modelA_all.csv`.

- **Model B** (α drawn independently per token, 200k rounds × 5 seeds): the landmark formula stays exact (error 0.06% at most) and a policy that adapts on information from before the round gains nothing (-0.23% to -0.00%). A per-token look-ahead oracle that sees each token's α before drafting it does gain 0.4–2.8% (see section 7).
- **Model C** (Markov regimes α ∈ {0.95, 0.4}, 20k rounds × 5 seeds): the adaptive policy that knows the state at round start beats the best fixed γ by 13.8% (s = 0.01), 11.8% (s = 0.05), 5.3% (s = 0.2); the per-token state oracle by 12.7%, 12.9%, 9.7%. The landmark formula with the stationary mean α = 0.675 *overestimates* throughput here (-8.3%, -7.5%, -4.8%).
- **Model D** (noisy System 1, κ = 10): the oracle gain of 12.6% survives as 100% at σ = 0, 72% at σ = 0.05, 19% at σ = 0.1, -79% at σ = 0.2 of the oracle gain (negative = worse than best fixed γ).

Combined table: `results/table2_variable_alpha.md`.

## 5. E3 — GPT-2 traces

Passages: prose: 100 used of 662 with ≥160 tokens (2183 candidates); maths: 100 used of 558 with ≥160 tokens (1319 candidates); code: 100 used of 133 with ≥160 tokens (164 candidates).

**Measured c** (64 tokens one at a time with KV cache, 5 repeats after 2 warm-ups):

| model | role | per-token time (ms) | c | 8-token verify ÷ 1-token step |
|---|---|---|---|---|
| distilgpt2 | draft | 5.88 ± 0.69 |  |  |
| gpt2-large | target | 47.72 ± 0.41 | 0.123 | 1.77 |
| gpt2-medium | target | 23.98 ± 0.43 | 0.245 | 1.30 |

**Is α constant?** Descriptive statistics of β_t (pooled over passages and positions):

| pair | type | mean | SD | q05 | median | q95 | lag-1 ACF | lag-5 ACF | lag-10 ACF | frac β>0.9 | frac β<0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| large | prose | 0.590 | 0.210 | 0.217 | 0.599 | 0.970 | 0.208 | 0.043 | 0.011 | 0.083 | 0.095 |
| large | maths | 0.601 | 0.236 | 0.171 | 0.615 | 0.972 | 0.182 | 0.090 | 0.001 | 0.120 | 0.120 |
| large | code | 0.604 | 0.270 | 0.142 | 0.613 | 0.989 | 0.200 | 0.111 | 0.058 | 0.194 | 0.157 |
| medium | prose | 0.613 | 0.193 | 0.273 | 0.619 | 0.969 | 0.189 | 0.030 | 0.001 | 0.082 | 0.062 |
| medium | maths | 0.627 | 0.217 | 0.227 | 0.643 | 0.972 | 0.163 | 0.091 | 0.018 | 0.117 | 0.084 |
| medium | code | 0.628 | 0.251 | 0.189 | 0.633 | 0.991 | 0.232 | 0.082 | 0.059 | 0.193 | 0.116 |

**Throughput on the test half at measured c** (tokens per target-step unit, mean ± SD over 5 replay seeds; γ in brackets in the CSV):

| pair | type | c | no_spec | fixed_4 | landmark | best_fixed | adaptive_oracle | adaptive_realistic | landmark_gamma | best_fixed_gamma | realistic_calibrator | realistic_recovery_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| large | code | 0.1232 | 1.000 ± 0.000 | 1.566 ± 0.017 | 1.602 ± 0.012 | 1.602 ± 0.012 | 1.722 ± 0.015 | 1.644 ± 0.012 | 3 | 3 | isotonic, lam=1.585 | 34.9 |
| large | maths | 0.1232 | 1.000 ± 0.000 | 1.538 ± 0.007 | 1.582 ± 0.009 | 1.582 ± 0.009 | 1.674 ± 0.008 | 1.606 ± 0.007 | 3 | 3 | isotonic, lam=1.580 | 26.2 |
| large | prose | 0.1232 | 1.000 ± 0.000 | 1.509 ± 0.007 | 1.556 ± 0.005 | 1.556 ± 0.005 | 1.622 ± 0.008 | 1.573 ± 0.006 | 3 | 3 | isotonic, lam=1.571 | 26.3 |
| medium | code | 0.2452 | 1.000 ± 0.000 | 1.225 ± 0.008 | 1.357 ± 0.006 | 1.357 ± 0.006 | 1.459 ± 0.011 | 1.383 ± 0.005 | 2 | 2 | isotonic, lam=1.367 | 25.4 |
| medium | maths | 0.2452 | 1.000 ± 0.000 | 1.222 ± 0.004 | 1.354 ± 0.004 | 1.354 ± 0.004 | 1.430 ± 0.005 | 1.365 ± 0.005 | 2 | 2 | logistic, lam=1.353 | 13.6 |
| medium | prose | 0.2452 | 1.000 ± 0.000 | 1.191 ± 0.010 | 1.329 ± 0.011 | 1.329 ± 0.011 | 1.379 ± 0.014 | 1.334 ± 0.010 | 2 | 2 | isotonic, lam=1.340 | 11.2 |

**Realistic System 1 recovery** = (realistic − best fixed) ÷ (oracle − best fixed), test half:

| pair | type | measured c | c=0.05 | c=0.1 | c=0.2 |
|---|---|---|---|---|---|
| large | prose | 26% | 10% | 22% | 12% |
| large | maths | 26% | 25% | 24% | 18% |
| large | code | 35% | 31% | 32% | 27% |
| medium | prose | 11% | 22% | 26% | 19% |
| medium | maths | 14% | 29% | 26% | 20% |
| medium | code | 25% | 41% | 29% | 25% |

Sensitivity table over c: `results/table3_gpt2_sensitivity.md`. Calibrator choice per cell: `results/e3_calibrators.csv`.

**Jensen check on real data** (test half, measured c, landmark γ): formula with mean β vs replay at the same γ.

The control column replays the same passages with every β_t set to the mean β (same uniforms), which isolates the effect of 128-token passages (rounds truncated at the end) from the effect of variable β.

| pair | type | γ | mean β | landmark pred | replay (real β) | replay (constant β control) | real − pred | real − control | lag-1 ACF |
|---|---|---|---|---|---|---|---|---|---|
| large | prose | 3 | 0.591 | 1.566 | 1.556 ± 0.005 | 1.565 ± 0.010 | -0.7% | -0.6% | 0.193 |
| large | maths | 3 | 0.606 | 1.604 | 1.582 ± 0.009 | 1.602 ± 0.013 | -1.3% | -1.2% | 0.209 |
| large | code | 3 | 0.617 | 1.630 | 1.602 ± 0.012 | 1.626 ± 0.013 | -1.7% | -1.4% | 0.172 |
| medium | prose | 2 | 0.612 | 1.333 | 1.329 ± 0.011 | 1.333 ± 0.009 | -0.3% | -0.4% | 0.178 |
| medium | maths | 2 | 0.632 | 1.363 | 1.354 ± 0.004 | 1.364 ± 0.009 | -0.6% | -0.7% | 0.180 |
| medium | code | 2 | 0.636 | 1.369 | 1.357 ± 0.006 | 1.371 ± 0.007 | -0.9% | -1.0% | 0.217 |

Replay with real β is above the landmark formula in 0 of 6 cells and above the constant-β control in 0 of 6 cells; every cell has positive lag-1 autocorrelation: True.

## 6. Anything unexpected

Each item states the observed result (from the files) and, separately, a **guess** at the cause.

- Model A reproduces every expected value in the brief to the printed precision (e.g. κ = 5: best fixed 2.783 (10), adaptive 3.619, gain 30.0%). Nothing unexpected there.
- Model B: the brief expected adaptive policies to gain nothing. Policies using only past information gain -0.00% at most, as expected, but the per-token look-ahead oracle gains up to 2.8%. *Guess:* that oracle knows the α of the very token it is about to draft, which is information about the future even when tokens are independent; the brief's claim holds for policies that cannot see it.
- Model C: the landmark formula with mean α *overestimates* the fixed-γ throughput by 8.3% at most, the opposite sign to Model A. *Guess:* in a token-level Markov stream, throughput is a harmonic-type average over positions (hard stretches cost more time per token), and rounds that start in the hard state are short, so time is weighted toward the hard regime. Model A's Jensen gain comes from drawing α per round, which weights each round equally. Persistence alone does not guarantee that the landmark formula underestimates.
- Model D: at σ = 0.2 the noisy threshold policy is worse than the best fixed γ (-10.0%). *Guess:* the rule plugs α̂ straight into α^k ≥ λc without shrinking noisy estimates toward the mean, so large positive errors trigger very long, wasted drafts. A Bayes-shrunk α̂ would likely do better; we did not change the design.
- E3 Jensen check: the proposition predicts replay above the formula when β is positively autocorrelated, but in 6 of 6 cells the real-β replay is *below* the constant-β control (by -1.4% to -0.4%), with lag-1 ACF 0.17–0.22. *Guess:* same mechanism as Model C: the autocorrelation is weak and short-lived (lag-5 ACF ≈ 0.03–0.11), so per-token variability acts like Model B (no gain), while low-β positions end rounds early and so take up more of the time.
- Verifying 8 tokens with gpt2-large takes 1.77× one cached step, so the 'verification costs one step' assumption is only approximate on this device.
- Measured c for distilgpt2 → gpt2-medium is 0.245, well above the 6/24 layer ratio. *Guess:* on mps with batch size 1, per-token time for these small models is dominated by kernel launch and Python overhead rather than FLOPs, so the draft costs a larger share than its size suggests.

## 7. Deviations

- Project lives in `/Users/tanish/Desktop/prism-research` (the folder this session was opened in); `~/prism-spec` is a symlink to it, so `cd ~/prism-spec && bash run_all.sh` works as specified.
- Machine is an Apple M3 with 8 GB RAM, not an M1 with 16 GB as the brief assumed.
- `legacy/check_math.py` and `legacy/check_tc.py` were not provided to this session (not found on disk), so E0 was written from the specification in the brief alone; `legacy/` is empty.
- `pytest` was added to requirements.txt to run the tests.
- E0 check 5 compares max S against 1 + 1e-9 instead of 1: on the diagonal α = c, S(1) = 1 exactly but floating point gives 1 + 2e-16 at 15 grid points (the raw count is recorded in e0_checks.csv).
- E0 check 1b's pass criterion: L1 below twice the expected L1 of exact sampling from p with the same n.
- Model A uses cell-centred grid points with exact Beta mass per cell (CDF differences), so densities that diverge at 0 or 1 (e.g. m = 0.9, κ = 5) stay finite.
- Model B: 'adaptive' is reported two ways: a history-based threshold policy (uses the previous round's mean α) and a per-token look-ahead oracle. The brief did not define which adaptive policy to use under independent α.
- Model C: two adaptive policies are reported: γ chosen from the state at round start (grid search over γ_easy ∈ 0..40, γ_hard ∈ 0..15 on a separate fit seed), and the per-token rule using the true state of each drafted position. Best fixed γ and λ are chosen on a separate fit seed, then evaluated on 5 seeds.
- Model D computes throughput as Σ E[N | true α, γ(α̂)] / Σ cost (accept draws integrated out exactly) over 200k sampled rounds per seed; λ is refit for each σ on a separate fit seed.
- Monte-Carlo λ fits use the fixed-point iteration; if it cycles, the visited λ with the highest training throughput is kept.
- E3 replay: each position's accept/reject is pre-drawn as u_t < β_t with per-seed uniforms shared by all policies (common random numbers). This is exact because each position is drafted at most once. A round never drafts past the end of the passage, so every passage produces exactly 128 tokens.
- E3 landmark policy uses the best integer γ under the landmark formula at the training-half mean β; the real γ* is in `e3_jensen.csv`.
- E3 calibrators are fitted per (pair, text type) on the training half; logistic regression uses soft labels (each row duplicated with label 1 weight β and label 0 weight 1 − β). The better calibrator is chosen by training-half throughput of the realistic policy.
- Model downloads used `HF_HUB_DISABLE_XET=1` because Hugging Face's Xet transfer endpoint returned 404.

## Follow-up

Produced by `e4_followup.py`, appended after the main report. Every number below is read from that run's output files (`e4_*.csv`, `e4_convexity_check.json`, `timing_verify.csv`, `table3_gpt2_wallclock.*`).

### F1. Model C: naive adaptive rule and hard-state round starts

Naive rule: γ = largest k ≤ 40 with α_state^k ≥ λc, using the state at round start as if it lasted the whole round; λ fitted by fixed-point iteration on the fit seed; evaluated on the same 5 seeds as E2 (best fixed γ reproduces `e2_modelC.csv` exactly: True).

| s | best fixed (γ) | naive adaptive | naive γ easy / hard | naive gain | expected | share of rounds starting hard (best fixed γ) | expected |
|---|---|---|---|---|---|---|---|
| 0.01 | 2.043 ± 0.037 (5) | 2.259 ± 0.054 | 40 / 2 | +10.6% | ≈+12% | 75.3% ± 1.3 | ≈77% |
| 0.05 | 2.062 ± 0.012 (5) | 2.114 ± 0.020 | 40 / 2 | +2.6% | ≈+2% | 74.3% ± 0.4 | ≈75% |
| 0.2 | 2.121 ± 0.012 (5) | 1.716 ± 0.014 | 40 / 2 | -19.1% | ≈-18% | 68.9% ± 0.4 | ≈69% |

- (a) Naive-rule gain **matches** the expected values (tolerance ±2 percentage points).
- (b) Hard-start share **matches** the expected values (tolerance ±2 percentage points). The stationary share of *positions* in the hard state is 50%, so rounds over-sample the hard regime.
- Cap flag: the naive rule's γ hits the cap of 40 at s = 0.01, 0.05, 0.2 (0.95^k ≥ λc holds up to k ≈ 43 at s = 0.01). So in the easy state the naive rule's γ is set by the cap, not by the rule.

### F2. Harmonic-mean prediction for Model C

At the landmark γ = 5 (stationary mean α = 0.675), H = (½/g(0.95) + ½/g(0.4))⁻¹ ÷ (γc + 1) with g(α) = Σ_(k=0..γ) α^k.

| s | landmark | H | simulated fixed-γ | H vs landmark | simulated vs landmark | simulated vs H |
|---|---|---|---|---|---|---|
| 0.01 | 2.229 | 2.022 | 2.043 ± 0.037 | -9.3% | -8.3% | +1.0% |
| 0.05 | 2.229 | 2.022 | 2.062 ± 0.012 | -9.3% | -7.5% | +1.9% |
| 0.2 | 2.229 | 2.022 | 2.121 ± 0.012 | -9.3% | -4.8% | +4.9% |

- H is 9.3% below the landmark at γ = 5: **matches** the expected ≈9.3% at γ = 5. H is the s → 0 limit (each regime long enough to act as its own stream). As s grows, the simulation moves from H toward the landmark.
- Convexity check 2g′² − g·g″ ≥ −1e-12 on [0, 1] (100,001 points, γ = 1…60): minimum -1.07e-14 at γ = 52, α = 0.4078 → **passes**. As an exact companion check, every integer coefficient of 2g′² − g·g″ is non-negative for γ = 1…60: True. Because (1/g)″ = (2g′² − g·g″)/g³, 1/g is convex, so by Jensen the harmonic-type average never exceeds the landmark.

### F3. Re-timing of verify passes (one target and the draft loaded at a time)

One cached target pass over m new tokens after a 32-token prefix, 20 repeats after 5 warm-ups. v(m) = mean time ÷ mean time at m = 1.

| m | large: v(m) mean (median) | large: ms | medium: v(m) mean (median) | medium: ms |
|---|---|---|---|---|
| 1 | 1.00 (1.00) | 53.5 | 1.00 (1.00) | 22.7 |
| 2 | 1.65 (1.59) | 88.1 | 1.21 (1.23) | 27.5 |
| 3 | 1.68 (1.68) | 89.7 | 1.22 (1.20) | 27.5 |
| 4 | 1.66 (1.67) | 88.8 | 1.33 (1.24) | 30.2 |
| 5 | 1.62 (1.64) | 86.7 | 1.27 (1.27) | 28.7 |
| 6 | 1.58 (1.55) | 84.5 | 1.34 (1.35) | 30.3 |
| 7 | 1.70 (1.74) | 90.9 | 1.30 (1.31) | 29.6 |
| 8 | 1.72 (1.74) | 91.8 | 1.32 (1.33) | 29.9 |
| 9 | 1.70 (1.73) | 91.0 | 1.34 (1.34) | 30.3 |

- large: available RAM 1.37 GB before / 0.56 GB after timing; swap used 3.94 → 1.61 GB. 8-token ratio from E3 in this same run (both targets loaded in turn, `timing.csv`) 1.77×; isolated now 1.72×.

- medium: available RAM 1.42 GB before / 2.06 GB after timing; swap used 1.67 → 1.45 GB. 8-token ratio from E3 in this same run (both targets loaded in turn, `timing.csv`) 1.30×; isolated now 1.32×.
- The earlier run's 4.55× (gpt2-large, m = 8) is not in these results files any more; neither this run's E3 timing (1.77×) nor the isolated timing (1.72×) reproduces it, which fits a transient effect such as memory pressure but does not prove it. Even in isolation only 1.4 GB RAM was free with gpt2-large loaded, and v(m) for gpt2-large is not monotone in m (a non-monotone v means noisy timing, not a real cost structure).

### F4. Wall-clock replay (Table 3, second version)

Round cost T = γc + v(γ + 1), with c from `timing.csv` and v(m) from F3 (v(1) = 1). γ is capped at 8 because v is measured only up to m = 9. Best fixed γ, λ and calibrators are re-selected on the training half under this cost. The adaptive rule drafts the k-th token while ∏β ≥ λ·Δ_k, where Δ_k = c + v(k+1) − v(k) is that token's marginal cost. Test half, mean ± SD over 5 replay seeds, in units of one cached target step.

- large: c = 0.123; T(γ) for γ = 0…8: 1.00, 1.77, 1.92, 2.03, 2.11, 2.20, 2.44, 2.58, 2.69
- medium: c = 0.245; T(γ) for γ = 0…8: 1.00, 1.46, 1.71, 2.07, 2.25, 2.56, 2.78, 3.04, 3.30

| pair | type | c | no_spec | fixed_4 | landmark | best_fixed | adaptive_oracle | adaptive_realistic | landmark_gamma | best_fixed_gamma | realistic_calibrator | realistic_recovery_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| large | code | 0.1232 | 1.000 ± 0.000 | 1.106 ± 0.012 | 1.083 ± 0.009 | 1.106 ± 0.012 | 1.200 ± 0.004 | 1.132 ± 0.005 | 3 | 4 | logistic, lam=1.000 | 28.1 |
| large | maths | 0.1232 | 1.000 ± 0.000 | 1.086 ± 0.005 | 1.069 ± 0.006 | 1.086 ± 0.005 | 1.152 ± 0.005 | 1.112 ± 0.004 | 3 | 4 | isotonic, lam=1.000 | 38.9 |
| large | prose | 0.1232 | 1.000 ± 0.000 | 1.066 ± 0.005 | 1.051 ± 0.003 | 1.066 ± 0.005 | 1.135 ± 0.004 | 1.070 ± 0.003 | 3 | 4 | isotonic, lam=1.080 | 6.3 |
| medium | code | 0.2452 | 1.000 ± 0.000 | 1.080 ± 0.007 | 1.186 ± 0.005 | 1.186 ± 0.005 | 1.300 ± 0.006 | 1.204 ± 0.002 | 2 | 2 | isotonic, lam=1.192 | 15.6 |
| medium | maths | 0.2452 | 1.000 ± 0.000 | 1.078 ± 0.004 | 1.184 ± 0.003 | 1.184 ± 0.003 | 1.264 ± 0.006 | 1.189 ± 0.003 | 2 | 2 | isotonic, lam=1.179 | 6.8 |
| medium | prose | 0.2452 | 1.000 ± 0.000 | 1.050 ± 0.009 | 1.162 ± 0.009 | 1.162 ± 0.009 | 1.214 ± 0.007 | 1.165 ± 0.008 | 2 | 2 | isotonic, lam=1.163 | 6 |

Change from the linear-cost Table 3 (best fixed γ and adaptive oracle, test half):

| pair | type | best fixed: linear → wall-clock | adaptive oracle: linear → wall-clock |
|---|---|---|---|
| large | prose | 1.556 (γ=3) → 1.066 (γ=4) | 1.622 → 1.135 |
| large | maths | 1.582 (γ=3) → 1.086 (γ=4) | 1.674 → 1.152 |
| large | code | 1.602 (γ=3) → 1.106 (γ=4) | 1.722 → 1.200 |
| medium | prose | 1.329 (γ=2) → 1.162 (γ=2) | 1.379 → 1.214 |
| medium | maths | 1.354 (γ=2) → 1.184 (γ=2) | 1.430 → 1.264 |
| medium | code | 1.357 (γ=2) → 1.186 (γ=2) | 1.459 → 1.300 |

- Slower than plain decoding (throughput < 1) under wall-clock cost: none of the policies in any pair/type cell. The re-selected best fixed γ is never slower than plain decoding.
- Best fixed γ hits the cap of 8 in 0 cell(s).
- λ fixed-point fits that cycled instead of converging (best visited λ kept): 5 of 18 — large/prose oracle (true beta) (λ = 1.000); large/maths isotonic (λ = 1.000); large/maths logistic (λ = 1.000); large/code logistic (λ = 1.000); medium/prose logistic (λ = 1.160).

### F5. Model D with posterior-mean shrinkage

The policy uses E[α | α̂] under the known Beta(7, 3) prior, computed on the 20,001-point α grid. Gaussian likelihood for interior α̂; the clipped endpoints 0.01 and 0.99 are treated as censored. λ is refit per σ on the fit seed; 200k rounds × 5 seeds. The raw-α̂ policy reproduces `e2_modelD.csv` exactly: True.

| σ | raw α̂: throughput | raw: share of oracle gain kept | shrunk: throughput | shrunk: share of oracle gain kept |
|---|---|---|---|---|
| 0 | 2.878 ± 0.002 | 100% | 2.878 ± 0.002 | 100% |
| 0.05 | 2.786 ± 0.002 | 72% | 2.797 ± 0.001 | 75% |
| 0.1 | 2.615 ± 0.003 | 19% | 2.702 ± 0.001 | 46% |
| 0.2 | 2.300 ± 0.003 | -79% | 2.596 ± 0.001 | 13% |

Oracle 2.877, best fixed 2.556 (exact, Model A, κ = 10). No expected values were given for this item.

### F6. Figures

No table that `figures.py` reads (e1_*, e2_modelA_main, e3_beta_stats, timing.csv, traces, e3_fig4_example) was changed by this follow-up; E4 writes only new files, so the figures were not regenerated.

E4 runtime: 87 s.

