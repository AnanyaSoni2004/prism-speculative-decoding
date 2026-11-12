"""Proposition 3 under Markov persistence (the paper's Model C).

Model: each token position has a hidden regime, easy (acceptance 0.95) or hard (0.40); after every
token the regime switches with probability s. A round with gamma drafts costs gamma*c + 1.

Part A (regime observed at the start of each round): a 2-state semi-Markov decision problem.
  - exact optimum: enumerate every policy (gamma_easy, gamma_hard), exact throughput from the
    stationary distribution of the round-start regime
  - threshold rule of Proposition 3: draft while A_k >= lambda*c, lambda = its own throughput
Part B (regime hidden; System 1 only sees accept/reject outcomes): belief-state problem on a grid.
  - optimum by Dinkelbach + relative value iteration
  - threshold rule on the belief's survival probabilities
  Both are then scored by the same long Monte Carlo simulation.
"""
import numpy as np

AL = np.array([0.95, 0.40])
GMAX = 40


def sw(s):
    return np.array([[1 - s, s], [s, 1 - s]])


def round_stats(b, g, s):
    """Start-of-round regime distribution b, g drafts. Returns E[N], survival A_0..A_g,
    and for each outcome k (number accepted) its probability and the next-round regime distribution."""
    S = sw(s)
    f = np.array(b, float)
    A, outs = [1.0], []
    for k in range(g):
        rej = f * (1 - AL)                       # rejected at draft k+1, regime of that position
        outs.append((rej.sum(), rej @ S))
        f = (f * AL) @ S
        A.append(f.sum())
    outs.append((f.sum(), f @ S))                # all accepted; bonus token, then switch
    EN = sum(A)
    return EN, np.array(A), outs


# ------------------------------------------------------------------ Part A: regime observed
def throughput_obs(gH, gL, s, c):
    EN, P = [], []
    for z, g in ((0, gH), (1, gL)):
        e = np.eye(2)[z]
        en, _, outs = round_stats(e, g, s)
        EN.append(en)
        P.append(sum(p * (nb / p if p > 0 else nb) for p, nb in outs))
    P = np.array(P)
    pi = np.array([P[1, 0], P[0, 1]]); pi = pi / pi.sum()     # stationary dist of 2-state chain
    T = np.array([gH * c + 1, gL * c + 1])
    return (pi @ np.array(EN)) / (pi @ T)


def threshold_obs(s, c):
    lam = 1.0
    for _ in range(200):
        g = []
        for z in (0, 1):
            _, A, _ = round_stats(np.eye(2)[z], GMAX, s)
            ok = np.where(A[1:] >= lam * c)[0]               # A is non-increasing
            g.append(int(ok[-1] + 1) if len(ok) else 0)
        new = throughput_obs(g[0], g[1], s, c)
        if abs(new - lam) < 1e-12:
            break
        lam = new
    return g, lam


def part_a(s, c):
    best = max(((throughput_obs(a, b, s, c), a, b) for a in range(GMAX + 1) for b in range(GMAX + 1)))
    fixed = max((throughput_obs(g, g, s, c), g) for g in range(GMAX + 1))
    g, lam = threshold_obs(s, c)
    return fixed, best, (lam, g[0], g[1])


# ------------------------------------------------------------------ Part B: regime hidden
NB = 401
GRID = np.linspace(0, 1, NB)                      # belief = P(easy) at the start of a round


def build(s, c, gmax=GMAX):
    """For every belief grid point and gamma: reward E[N], time, and transitions (interpolated)."""
    R = np.zeros((NB, gmax + 1)); A_all = np.zeros((NB, gmax + 1))
    idx = np.zeros((NB, gmax + 1, gmax + 1, 2), int); w = np.zeros((NB, gmax + 1, gmax + 1, 2))
    for i, p in enumerate(GRID):
        _, A, outs = round_stats([p, 1 - p], gmax, s)
        A_all[i] = A
        # outcomes for a smaller g share the same rejection branches; only the "all accepted" one differs
        f = np.array([p, 1 - p]); S = sw(s)
        for g in range(gmax + 1):
            R[i, g] = A[: g + 1].sum()
            branches = [(outs[k][0], outs[k][1]) for k in range(g)] + [(f.sum(), f @ S)]
            for k, (pr, nb) in enumerate(branches):
                pe = nb[0] / pr if pr > 1e-300 else 0.5
                x = pe * (NB - 1); lo = min(int(np.floor(x)), NB - 2)
                idx[i, g, k] = (lo, lo + 1); w[i, g, k] = (pr * (lo + 1 - x), pr * (x - lo))
            f = (f * AL) @ S
    return R, A_all, idx, w


def solve_hidden(R, idx, w, c, gmax=GMAX):
    T = np.arange(gmax + 1) * c + 1.0
    lam, h = 1.0, np.zeros(NB)
    for _ in range(60):                            # Dinkelbach outer loop
        for _ in range(4000):                      # relative value iteration on reward N - lam*T
            Q = R - lam * T[None, :] + (w * h[idx]).sum((2, 3))
            hn = Q.max(1); gain = hn[NB // 2]; hn = hn - gain
            if np.abs(hn - h).max() < 1e-11:
                h = hn; break
            h = hn
        pol = Q.argmax(1)
        if abs(gain) < 1e-10:
            break
        lam += gain / (pol.mean() * c + 1)          # Newton-like update of the ratio
    return pol, lam


def threshold_hidden(A_all, lam, c):
    pol = np.zeros(NB, int)
    for i in range(NB):
        ok = np.where(A_all[i, 1:] >= lam * c)[0]
        pol[i] = ok[-1] + 1 if len(ok) else 0
    return pol


def simulate(pol, s, c, rounds=1_500_000, seed=0):
    """Monte Carlo score of a belief policy (nearest grid point), true hidden regime simulated."""
    rng = np.random.default_rng(seed)
    S = sw(s); z = 0; b = np.array([0.5, 0.5]); tok = tim = 0.0
    U = rng.random((rounds, 2 * GMAX + 4)); u = 0
    for r in range(rounds):
        g = pol[int(round(b[0] * (NB - 1)))]
        row = U[r]; j = 0
        f = b.copy(); k = 0
        while k < g:
            if row[j] < AL[z]:
                f = (f * AL) @ S; k += 1
                z = z if row[j + 1] >= s else 1 - z
                j += 2
            else:
                f = f * (1 - AL); break
        f = f @ S if k < g else f @ S                # token at the stop position, then switch
        z = z if row[j + 1] >= s else 1 - z
        b = f / f.sum()
        tok += k + 1; tim += g * c + 1
    return tok / tim


def part_b(s, c):
    R, A_all, idx, w = build(s, c)
    opt_pol, lam_opt = solve_hidden(R, idx, w, c)
    lam = 1.0
    for _ in range(40):                             # threshold rule with self-consistent lambda (model-based)
        thr_pol = threshold_hidden(A_all, lam, c)
        # evaluate on the grid chain: gain of this fixed policy by the same recursion
        T = thr_pol * c + 1.0; h = np.zeros(NB); l2 = lam
        for _ in range(60):
            for _ in range(4000):
                q = R[np.arange(NB), thr_pol] - l2 * T + (w[np.arange(NB), thr_pol] * h[idx[np.arange(NB), thr_pol]]).sum((1, 2))
                gain = q[NB // 2]; hn = q - gain
                if np.abs(hn - h).max() < 1e-11:
                    h = hn; break
                h = hn
            if abs(gain) < 1e-10:
                break
            l2 += gain / T.mean()
        if abs(l2 - lam) < 1e-9:
            break
        lam = l2
    fixed = max((simulate(np.full(NB, g), s, c, 400_000), g) for g in range(1, 13))
    return fixed, simulate(opt_pol, s, c), simulate(thr_pol, s, c), opt_pol, thr_pol, lam_opt, lam


if __name__ == "__main__":
    for c in (0.05, 0.2):
        print(f"\n################ c = {c}")
        print("Part A: regime observed at round start (exact)")
        print("   s     best fixed (g)     threshold rule (gE,gH)      exact optimum (gE,gH)      gap")
        for s in (0.01, 0.05, 0.2, 0.5):
            fx, best, th = part_a(s, c)
            print(f"  {s:<5} {fx[0]:.4f} ({fx[1]:>2})       {th[0]:.4f} ({th[1]:>2},{th[2]:>2})            "
                  f"{best[0]:.4f} ({best[1]:>2},{best[2]:>2})        {100 * (best[0] / th[0] - 1):.3f}%")
        print("Part B: regime hidden, belief from accept/reject outcomes (Monte Carlo, 1.5M rounds)")
        print("   s     best fixed (g)     threshold rule     optimum (belief MDP)     gap     drafts at belief 0 / 0.5 / 1: threshold | optimum")
        for s in (0.01, 0.05, 0.2):
            fx, o, t, op, tp, lo, lt = part_b(s, c)
            print(f"  {s:<5} {fx[0]:.4f} ({fx[1]:>2})       {t:.4f}             {o:.4f}                {100 * (o / t - 1):+.2f}%   "
                  f"{tp[0]},{tp[NB // 2]},{tp[-1]} | {op[0]},{op[NB // 2]},{op[-1]}")
