# Theory notes (12 Nov 2026)

## 1. Proposition 2′: 1/g is convex for every γ (replaces the "checked for γ ≤ 60" step)

Let g(α) = Σ_{k=0}^{γ} α^k. Since (1/g)'' = (2g'² − g·g'') / g³ and g > 0, it is enough to show
2g'² − g·g'' ≥ 0 on [0, 1].

**Identity.** For every integer γ ≥ 1,

    2g'(α)² − g(α)·g''(α) = (γ+1) · Σ_{s=1}^{γ} s(γ+1−s) · α^{γ+s−2}.

Every coefficient on the right is positive, so 1/g is convex on [0, 1] (strictly on (0, 1]).

**Proof.** Write g'² = Σ_{i,j} ij·α^{i+j−2} and g·g'' = Σ_{i,j} j(j−1)·α^{i+j−2}, with 0 ≤ i, j ≤ γ.
The coefficient of α^{n−2} in 2g'² − g·g'' is the sum over pairs with i + j = n of 2ij − j(j−1).
Symmetrising in (i, j) and writing d = i − j (so ij = (n² − d²)/4 and i² + j² = (n² + d²)/2),
each pair contributes

    n²/4 + n/2 − 3d²/4.

The valid pairs have d = −m, −m+2, …, m, where m = n for n ≤ γ and m = 2γ − n for n > γ.
There are m + 1 of them and Σ d² = m(m+1)(m+2)/3, so the coefficient equals

    (m+1)/4 · [ n(n+2) − m(m+2) ].

* If n ≤ γ then m = n and the coefficient is 0.
* If n > γ then m = 2γ − n < n and n(n+2) − m(m+2) = (n−m)(n+m+2) = 4(n−γ)(γ+1),
  so the coefficient is (2γ−n+1)(n−γ)(γ+1) > 0.

Putting s = n − γ gives the identity. ∎

Checked symbolically (sympy) for γ = 1, …, 80: the two sides agree exactly.

Remark. As γ → ∞, g → 1/(1−α) and 1/g → 1 − α is linear; the identity shows this directly,
because the lowest power on the right is α^{γ−1}, which vanishes on [0, 1) as γ grows.

## 2. Novelty check for the closed-form optimal γ (Proposition 1c)

The Lambert-W closed form is already published. It must be cited, and presented as a re-derivation.

* C. Zheng and T. Yang, "Communication-Efficient Collaborative LLM Inference via Distributed
  Speculative Decoding", arXiv:2509.04576. Theorem 2 (Eq. 21) gives the optimal draft length with
  W_{-1} for S(γ) = (1 − α^{γ+1}) / ((1 + γL)(1 − α)), where L = b + c adds a communication cost b
  to the draft cost c. With b = 0 this is the same objective as ours.
* A. Bozorgkhoo and I. Molybog, "Speculative Decoding Scaling Laws (SDSL)", arXiv:2603.11053.
  Appendix C (Eq. 13) gives γ_opt with W_{-1}; Section 3.2 (Eq. 4) gives the throughput at the optimum.

To check by hand before submission: Leviathan et al. (2023) state a corollary that an improvement
exists when α > c, with factor at least (1 + α)/(1 + c). If so, the "if" half of Proposition 1(b)
is theirs and only the "only if" half (via quasi-concavity) is ours.

What remains ours after this check: the two-sided bound and its sign flip (Propositions 2 and 2′),
the exact-replay identity, and the measured results on own-text traces.

## 3. Proposition 3 under Markov persistence (Model C): the threshold rule is not optimal

Setup: two regimes (acceptance 0.95 / 0.40), switch probability s after every token, round cost γc + 1.
Code: `prop3_markov.py`; output: `prop3_results.txt`.

**Regime observed at the start of each round (exact, 2-state semi-Markov decision problem).**
Throughput; (drafts in easy, drafts in hard):

| c | s | best fixed γ | threshold rule A_k ≥ λc | exact optimum | gap |
|---|---|---|---|---|---|
| 0.05 | 0.01 | 2.027 (5) | 2.273 (35, 2) | 2.323 (19, 2) | 2.2% |
| 0.05 | 0.05 | 2.046 (5) | 2.258 (22, 2) | 2.289 (14, 2) | 1.4% |
| 0.05 | 0.2 | 2.119 (5) | 2.226 (10, 3) | 2.231 (9, 3) | 0.2% |
| 0.05 | 0.5 | 2.229 (5) | 2.249 (6, 4) | 2.249 (6, 4) | 0.0% |
| 0.2 | 0.01 | 1.442 (2) | 1.609 (18, 1) | 1.653 (9, 1) | 2.7% |
| 0.2 | 0.05 | 1.447 (2) | 1.601 (12, 1) | 1.634 (7, 1) | 2.0% |
| 0.2 | 0.2 | 1.467 (2) | 1.591 (6, 1) | 1.602 (5, 1) | 0.7% |
| 0.2 | 0.5 | 1.524 (3) | 1.580 (3, 1) | 1.580 (3, 1) | 0.0% |

At s = 0.5 the next regime is independent of the past, rounds are independent, and the rule is exactly
optimal, as Proposition 3 says. With persistence it over-drafts in easy text.

The optimum is not of the single-threshold form at all: at c = 0.05, s = 0.01 it needs
A_19(easy) ≥ θ > A_20(easy) (θ near 0.3) and A_2(hard) ≥ θ > A_3(hard) (θ ≤ 0.16) at once.

Reading: λc is the opportunity cost of one draft step. With independent rounds that cost is the global
throughput λ. With persistence it is the local one, which is higher inside an easy stretch, so drafting
should stop earlier there. Check: for constant α = 0.95, c = 0.05 the best fixed γ is about 19–21,
which is what the optimum uses in the easy regime when s = 0.01.

**Regime hidden (System 1 sees only accept/reject outcomes; belief-state problem on a 401-point grid,
both policies scored by the same 1.5M-round simulation).** The optimum beats the threshold rule by
3.2%, 2.6%, 1.0% (c = 0.05; s = 0.01, 0.05, 0.2) and 3.8%, 2.0%, 0.3% (c = 0.2). At c = 0.05, s = 0.2
the threshold rule (2.110) is below the best fixed γ (2.119).

Cross-checks: the exact observed-regime values agree with an independent simulation to 0.1–0.2%
((5,5): 2.022 vs 2.027; (35,2): 2.272 vs 2.273; (19,2): 2.321 vs 2.323). Best-fixed values are close
to, but not identical with, the paper's Table 1 (2.027 / 2.046 / 2.119 here vs 2.043 / 2.062 / 2.121),
so the paper's Model C differs in some small detail from this implementation.

Consequence for the paper: state Proposition 3 under (A1′) only. Under persistence the rule is a
heuristic that loses up to about 4% to the optimum in this model, and the fix is a state-dependent
threshold.
