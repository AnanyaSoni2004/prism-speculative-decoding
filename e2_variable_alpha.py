"""E2: when alpha changes across the text.

Model A  alpha ~ Beta per round, persistent within a round (exact, 20,001-point grid)
Model B  alpha independent per token (Monte Carlo control)
Model C  two-state Markov regimes (Monte Carlo replay)
Model D  noisy System-1 estimate of alpha in Model A (Monte Carlo)
"""
import numpy as np
import pandas as pd

from prism.core import (
    GAMMA_MAX,
    best_integer_gamma,
    beta_grid,
    expected_tokens,
    fit_lambda_fixed_point,
    fit_lambda_persistent,
    next_reject,
    per_token_gamma,
    replay_rounds,
    speedup,
    threshold_policy,
)
from prism.util import RESULTS, SEEDS, Timer, log_seeds

KAPPAS = [np.inf, 50, 20, 10, 5]
G = np.arange(GAMMA_MAX + 1)


# --------------------------------------------------------------------------- Model A
def model_a(m, kappa, c):
    if np.isinf(kappa):
        x, w = np.array([m]), np.array([1.0])
    else:
        x, w = beta_grid(m, kappa, 20001)
    mom = np.array([np.sum(w * x**k) for k in range(GAMMA_MAX + 1)])
    true = np.cumsum(mom) / (G * c + 1.0)  # true throughput of every fixed gamma
    kL, sL = best_integer_gamma(m, c)
    kB = int(np.argmax(true))
    lam, iters = fit_lambda_persistent(x, w, c)
    g = threshold_policy(x, lam, c)
    adapt = np.sum(w * expected_tokens(x, g)) / np.sum(w * (g * c + 1.0))
    sd = 0.0 if np.isinf(kappa) else np.sqrt(m * (1 - m) / (kappa + 1))
    return dict(
        m=m, c=c, kappa=kappa, sd_alpha=sd,
        landmark_pred=sL, landmark_gamma=kL,
        fixed_true_at_landmark_gamma=true[kL],
        best_fixed_true=true[kB], best_fixed_gamma=kB,
        adaptive_oracle=adapt, adaptive_lambda=lam, lambda_iters=iters,
        adaptive_mean_gamma=float(np.sum(w * g)),
        adaptive_mass_at_cap=float(np.sum(w[g == GAMMA_MAX])),
        gain_pct=100 * (adapt / true[kB] - 1),
        landmark_error_pct=100 * (true[kL] / sL - 1),
        best_fixed_hits_cap=kB == GAMMA_MAX,
    )


# --------------------------------------------------------------------------- Model B
def model_b(m, kappa, c, n_rounds, fit_seed, eval_seeds):
    a_, b_ = m * kappa, (1 - m) * kappa

    def draw(seed):
        rng = np.random.default_rng(seed)
        al = rng.beta(a_, b_, size=(n_rounds, GAMMA_MAX))
        acc = rng.uniform(size=(n_rounds, GAMMA_MAX)) < al
        fr = np.where(acc.all(1), GAMMA_MAX, np.argmin(acc, axis=1))  # accepted run length
        return al, fr

    def fixed_thr(fr):
        return np.array([np.mean(np.minimum(fr, g) + 1) / (g * c + 1) for g in G])

    def lookahead_gamma(al, lam):
        cl = np.cumsum(np.log(al), axis=1)  # log prod_{j<=k}
        return np.sum(cl >= np.log(lam * c), axis=1)  # products are decreasing -> count = largest k

    def lookahead_thr(al, fr, lam):
        g = lookahead_gamma(al, lam)
        return np.mean(np.minimum(fr, g) + 1) / np.mean(g * c + 1)

    def history_gamma(al, lam):
        # uses the previous round's mean alpha (information available before the round starts)
        prev = np.concatenate([[m], al[:-1].mean(1)])
        return threshold_policy(prev, lam, c)

    def history_thr(al, fr, lam):
        g = history_gamma(al, lam)
        return np.mean(np.minimum(fr, g) + 1) / np.mean(g * c + 1)

    al_f, fr_f = draw(fit_seed)
    gB = int(np.argmax(fixed_thr(fr_f)))
    lam_la, _, _ = fit_lambda_fixed_point(lambda l: lookahead_thr(al_f, fr_f, l))
    lam_hi, _, _ = fit_lambda_fixed_point(lambda l: history_thr(al_f, fr_f, l))
    kL, sL = best_integer_gamma(m, c)
    res = dict(fixed_at_landmark=[], best_fixed=[], history_adaptive=[], lookahead_oracle=[])
    for s in eval_seeds:
        al, fr = draw(s)
        ft = fixed_thr(fr)
        res["fixed_at_landmark"].append(ft[kL])
        res["best_fixed"].append(ft[gB])
        res["history_adaptive"].append(history_thr(al, fr, lam_hi))
        res["lookahead_oracle"].append(lookahead_thr(al, fr, lam_la))
    out = dict(model="B", m=m, c=c, kappa=kappa, sd_alpha=np.sqrt(m * (1 - m) / (kappa + 1)),
               landmark_pred=sL, landmark_gamma=kL, best_fixed_gamma=gB)
    for k, v in res.items():
        out[k + "_mean"] = float(np.mean(v))
        out[k + "_sd"] = float(np.std(v, ddof=1))
    out["landmark_error_pct"] = 100 * (out["fixed_at_landmark_mean"] / sL - 1)
    out["history_gain_pct"] = 100 * (out["history_adaptive_mean"] / out["best_fixed_mean"] - 1)
    out["lookahead_gain_pct"] = 100 * (out["lookahead_oracle_mean"] / out["best_fixed_mean"] - 1)
    return out


# --------------------------------------------------------------------------- Model C
def model_c(s, c, n_rounds, fit_seed, eval_seeds, a_easy=0.95, a_hard=0.4):
    L = n_rounds * (GAMMA_MAX + 2) + 10

    def stream(seed):
        rng = np.random.default_rng(seed)
        flips = rng.uniform(size=L) < s
        state0 = rng.uniform() < 0.5
        easy = np.logical_xor(state0, np.cumsum(flips) % 2 == 1)
        alpha = np.where(easy, a_easy, a_hard)
        acc = rng.uniform(size=L) < alpha
        return easy, alpha, next_reject(acc)

    def thr(nz, gam):
        tok, cost = replay_rounds(nz, gam, c, n_rounds)
        return tok / cost

    easy_f, alpha_f, nz_f = stream(fit_seed)
    fixed_f = [thr(nz_f, g) for g in G]
    gB = int(np.argmax(fixed_f))
    # adaptive A: gamma chosen from the state at the start of the round
    best = (-1, 0, 0)
    for ge in range(GAMMA_MAX + 1):
        for gh in range(16):
            v = thr(nz_f, np.where(easy_f, ge, gh))
            if v > best[0]:
                best = (v, ge, gh)
    _, ge, gh = best
    # adaptive B: per-token rule using the true state of each drafted position
    lam, it, conv = fit_lambda_fixed_point(lambda l: thr(nz_f, per_token_gamma(alpha_f, l, c)))
    m_stat = 0.5 * (a_easy + a_hard)
    kL, sL = best_integer_gamma(m_stat, c)
    res = dict(fixed_at_landmark=[], best_fixed=[], adaptive_state=[], adaptive_per_token=[])
    for sd in eval_seeds:
        easy, alpha, nz = stream(sd)
        res["fixed_at_landmark"].append(thr(nz, kL))
        res["best_fixed"].append(thr(nz, gB))
        res["adaptive_state"].append(thr(nz, np.where(easy, ge, gh)))
        res["adaptive_per_token"].append(thr(nz, per_token_gamma(alpha, lam, c)))
    out = dict(model="C", switch_prob=s, c=c, alpha_easy=a_easy, alpha_hard=a_hard, stationary_mean_alpha=m_stat,
               landmark_pred=sL, landmark_gamma=kL, best_fixed_gamma=gB, state_gamma_easy=ge,
               state_gamma_hard=gh, per_token_lambda=lam, lambda_converged=conv)
    for k, v in res.items():
        out[k + "_mean"] = float(np.mean(v))
        out[k + "_sd"] = float(np.std(v, ddof=1))
    out["state_gain_pct"] = 100 * (out["adaptive_state_mean"] / out["best_fixed_mean"] - 1)
    out["per_token_gain_pct"] = 100 * (out["adaptive_per_token_mean"] / out["best_fixed_mean"] - 1)
    out["landmark_error_pct"] = 100 * (out["fixed_at_landmark_mean"] / sL - 1)
    out["best_fixed_hits_cap"] = gB == GAMMA_MAX
    return out


# --------------------------------------------------------------------------- Model D
def model_d(m, kappa, c, sigma, n_rounds, fit_seed, eval_seeds, a_ref):
    a_, b_ = m * kappa, (1 - m) * kappa

    def draw(seed):
        rng = np.random.default_rng(seed)
        al = rng.beta(a_, b_, size=n_rounds)
        ah = np.clip(al + rng.normal(0.0, sigma, size=n_rounds), 0.01, 0.99) if sigma > 0 else al
        return al, ah

    def thr(al, ah, lam):  # throughput uses the true alpha (Rao-Blackwellised over accepts)
        g = threshold_policy(ah, lam, c)
        return np.sum(expected_tokens(al, g)) / np.sum(g * c + 1.0)

    al_f, ah_f = draw(fit_seed)
    lam, it, conv = fit_lambda_fixed_point(lambda l: thr(al_f, ah_f, l))
    vals = [thr(*draw(s), lam) for s in eval_seeds]
    mean = float(np.mean(vals))
    bf, orc = a_ref["best_fixed_true"], a_ref["adaptive_oracle"]
    return dict(model="D", m=m, c=c, kappa=kappa, sigma=sigma, lam=lam, lambda_converged=conv,
                throughput_mean=mean, throughput_sd=float(np.std(vals, ddof=1)),
                best_fixed_true=bf, adaptive_oracle_exact=orc,
                gain_pct=100 * (mean / bf - 1), oracle_gain_pct=100 * (orc / bf - 1),
                gain_survival_pct=100 * (mean - bf) / (orc - bf))


def main():
    fit, ev = SEEDS["e2_fit"], SEEDS["e2_eval"]
    log_seeds("e2", {"fit": fit, "eval": ev})

    rowsA = [model_a(m, k, c) for m in [0.5, 0.7, 0.8, 0.9] for c in [0.05, 0.1, 0.2] for k in KAPPAS]
    A = pd.DataFrame(rowsA)
    A.to_csv(RESULTS / "e2_modelA_all.csv", index=False)
    main_a = A[(A.m == 0.7) & (A.c == 0.05)]
    main_a.to_csv(RESULTS / "e2_modelA_main.csv", index=False)
    cols = ["kappa", "sd_alpha", "landmark_pred", "landmark_gamma", "best_fixed_true", "best_fixed_gamma",
            "adaptive_oracle", "gain_pct", "landmark_error_pct", "best_fixed_hits_cap", "adaptive_mass_at_cap"]
    print("Model A main (m=0.7, c=0.05)\n", main_a[cols].round(4).to_string(index=False))
    print("cap hits (any setting):", int(A.best_fixed_hits_cap.sum()), "best-fixed;",
          int((A.adaptive_mass_at_cap > 0).sum()), "settings with adaptive mass at cap")

    rowsB = [model_b(0.7, k, 0.05, 200_000, fit, ev) for k in [50, 20, 10, 5]]
    B = pd.DataFrame(rowsB)
    B.to_csv(RESULTS / "e2_modelB.csv", index=False)
    print("\nModel B\n", B[["kappa", "landmark_pred", "fixed_at_landmark_mean", "best_fixed_mean", "best_fixed_gamma",
                            "history_gain_pct", "lookahead_gain_pct", "landmark_error_pct"]].round(4).to_string(index=False))

    rowsC = [model_c(s, 0.05, 20_000, fit, ev) for s in [0.01, 0.05, 0.2]]
    C = pd.DataFrame(rowsC)
    C.to_csv(RESULTS / "e2_modelC.csv", index=False)
    print("\nModel C\n", C[["switch_prob", "best_fixed_gamma", "best_fixed_mean", "best_fixed_sd", "adaptive_state_mean",
                            "adaptive_state_sd", "state_gain_pct", "adaptive_per_token_mean", "per_token_gain_pct",
                            "landmark_error_pct"]].round(4).to_string(index=False))

    ref = main_a[main_a.kappa == 10].iloc[0].to_dict()
    rowsD = [model_d(0.7, 10, 0.05, sg, 200_000, fit, ev, ref) for sg in [0.0, 0.05, 0.1, 0.2]]
    D = pd.DataFrame(rowsD)
    D.to_csv(RESULTS / "e2_modelD.csv", index=False)
    print("\nModel D\n", D[["sigma", "throughput_mean", "throughput_sd", "gain_pct", "gain_survival_pct"]].round(4).to_string(index=False))


if __name__ == "__main__":
    with Timer("e2"):
        main()
