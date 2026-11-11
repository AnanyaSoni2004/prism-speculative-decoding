"""E0: every formula checked against simulation. Writes results/e0_checks.csv."""
import numpy as np
import pandas as pd

from prism.core import beta_moments, expected_tokens, gamma_star, speedup
from prism.util import RESULTS, SEEDS, Timer, log_seeds

rows = []


def add(check, formula, simulated, passed, tol_note=""):
    rows.append(dict(check=check, formula_value=float(formula), simulated_value=float(simulated),
                     abs_error=float(abs(formula - simulated)), pass_=bool(passed), tolerance=tol_note))


def main():
    seed = SEEDS["e0"]
    log_seeds("e0", seed)
    rng = np.random.default_rng(seed)
    V = 50

    # 1. distribution preserved (exact law) + 2. beta identity
    max_out_err = 0.0
    max_beta_err = 0.0
    for _ in range(1000):
        p = rng.dirichlet(np.full(V, 0.5))
        q = rng.dirichlet(np.full(V, 0.5))
        mn = np.minimum(p, q)
        beta = mn.sum()
        r = np.maximum(0.0, p - q) / (1.0 - beta) if beta < 1 else np.zeros(V)
        out = mn + (1.0 - beta) * r
        max_out_err = max(max_out_err, np.abs(out - p).max())
        max_beta_err = max(max_beta_err, abs(beta - (1 - 0.5 * np.abs(p - q).sum())))
    add("1a exact output law: max|out - p| over 1000 pairs", 0.0, max_out_err, max_out_err < 1e-12, "< 1e-12")

    # 1b. sampled draws
    p = rng.dirichlet(np.full(V, 0.5))
    q = rng.dirichlet(np.full(V, 0.5))
    n = 200_000
    x = rng.choice(V, size=n, p=q)
    acc = rng.uniform(size=n) < np.minimum(1.0, p[x] / q[x])
    r = np.maximum(0.0, p - q)
    r /= r.sum()
    x = np.where(acc, x, rng.choice(V, size=n, p=r))
    emp = np.bincount(x, minlength=V) / n
    l1 = np.abs(emp - p).sum()
    noise = np.sum(np.sqrt(2 * p * (1 - p) / (np.pi * n)))  # expected L1 of exact sampling from p
    add("1b sampled draws (200k): L1(empirical, p); formula = expected L1 of exact sampling",
        noise, l1, l1 < 2 * noise, "< 2x expected sampling L1")
    add("2 beta = 1 - 0.5||p-q||_1: max error over 1000 pairs", 0.0, max_beta_err, max_beta_err < 1e-12, "< 1e-12")

    # 3. E[N] by simulation, independent accepts
    for a in [0.5, 0.7, 0.8, 0.9]:
        for g in [1, 3, 5, 10]:
            u = rng.uniform(size=(200_000, g)) < a
            N = np.where(u.all(1), g, np.argmin(u, axis=1)) + 1
            se = N.std() / np.sqrt(len(N))
            f = expected_tokens(a, g)
            add(f"3 E[N] alpha={a} gamma={g} (200k rounds)", f, N.mean(), abs(f - N.mean()) < 4 * se,
                f"< 4 SE ({4*se:.4f})")

    # 4. optimal gamma via Lambert W vs grid search
    grid = np.arange(0.0, 100.0, 1e-4)
    for a, c in [(0.6, 0.05), (0.8, 0.05), (0.8, 0.2), (0.9, 0.02)]:
        gs = gamma_star(a, c)
        s = speedup(a, grid, c)
        gg = grid[np.argmax(s)]
        add(f"4 gamma* alpha={a} c={c}: Lambert W vs real grid (step 1e-4)", gs, gg, abs(gs - gg) <= 1e-4, "<= 1e-4")
        ints = np.arange(0, 101)
        si = speedup(a, ints, c)
        bi = int(ints[np.argmax(si)])
        cand = [int(np.floor(gs)), int(np.ceil(gs))]
        best_of_cand = max(cand, key=lambda k: speedup(a, k, c))
        add(f"4 best integer gamma alpha={a} c={c}: floor/ceil of gamma* vs integer search",
            best_of_cand, bi, best_of_cand == bi, "exact")
        add(f"4 best integer speedup alpha={a} c={c}", speedup(a, best_of_cand, c), si.max(),
            abs(speedup(a, best_of_cand, c) - si.max()) < 1e-12, "< 1e-12")
    gs = gamma_star(0.8, 0.05)
    add("4 known value: gamma*(0.8, 0.05) = 7.8567", 7.8567, gs, abs(gs - 7.8567) < 5e-5, "< 5e-5")
    add("4 known value: S at best integer (0.8, 0.05) = 3.092", 3.092, speedup(0.8, 8, 0.05),
        abs(speedup(0.8, 8, 0.05) - 3.092) < 5e-4, "< 5e-4")

    # 5. speculation helps iff alpha > c
    vals = np.arange(1, 100) / 100.0
    gam = np.arange(1, 401)
    disagree = raw = 0
    for a in vals:
        best = speedup(a, gam, vals[:, None]).max(axis=1)  # over gamma for every c
        disagree += int(np.sum((best > 1.0 + 1e-9) != (a > vals)))
        raw += int(np.sum((best > 1.0) != (a > vals)))
    add("5 max_{gamma>=1} S > 1 iff alpha > c: disagreeing points on 99x99 grid (S>1+1e-9)", 0, disagree,
        disagree == 0, f"== 0; raw S>1 test gives {raw}, all on alpha=c where S(1)=1 up to 1e-15 rounding")

    # 6. Jensen proposition
    m, kappa, g = 0.7, 5.0, 6
    a_, b_ = m * kappa, (1 - m) * kappa
    exact = beta_moments(a_, b_, g).sum()
    add("6 known value: exact E[N] m=0.7 kappa=5 gamma=6 = 3.4894", 3.4894, exact, abs(exact - 3.4894) < 5e-5, "< 5e-5")
    n = 2_000_000
    N_sum = 0.0
    N_sq = 0.0
    for chunk in range(4):  # chunked to bound memory
        al = rng.beta(a_, b_, size=n // 4)
        u = rng.uniform(size=(n // 4, g)) < al[:, None]
        N = (np.where(u.all(1), g, np.argmin(u, axis=1)) + 1).astype(float)
        N_sum += N.sum()
        N_sq += (N**2).sum()
    mean = N_sum / n
    se = np.sqrt(N_sq / n - mean**2) / np.sqrt(n)
    add("6i persistent alpha: simulation (2M rounds) vs exact sum E[alpha^k]", exact, mean,
        abs(mean - exact) < 4 * se, f"< 4 SE ({4*se:.4f})")
    land = expected_tokens(m, g)
    add("6ii exact E[N] >= landmark at alpha=m (formula=landmark 3.0588)", land, exact, exact >= land, "exact >= landmark")
    N_sum = N_sq = 0.0
    for chunk in range(4):
        al = rng.beta(a_, b_, size=(n // 4, g))
        u = rng.uniform(size=(n // 4, g)) < al
        N = (np.where(u.all(1), g, np.argmin(u, axis=1)) + 1).astype(float)
        N_sum += N.sum()
        N_sq += (N**2).sum()
    mean = N_sum / n
    se = np.sqrt(N_sq / n - mean**2) / np.sqrt(n)
    add("6iii per-token independent alpha: simulation (2M) vs landmark 3.0588", land, mean,
        abs(mean - land) < 4 * se, f"< 4 SE ({4*se:.4f})")

    df = pd.DataFrame(rows).rename(columns={"pass_": "pass"})
    df.to_csv(RESULTS / "e0_checks.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(df[["check", "formula_value", "simulated_value", "abs_error", "pass"]].to_string(index=False))
    print(f"\nE0: {df['pass'].sum()}/{len(df)} checks passed")


if __name__ == "__main__":
    with Timer("e0"):
        main()
