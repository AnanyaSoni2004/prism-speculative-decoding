"""Core formulas, policies and the replay simulator for the PRISM speculative-decoding study.

Notation (shared by every script):
    alpha  acceptance probability of one draft token
    gamma  number of draft tokens per round (integer, 0..GAMMA_MAX)
    c      cost of one draft step / cost of one target step
    E[N]   expected tokens per round (accepted drafts + 1)
    S      speedup = tokens per unit time; one round costs gamma*c + 1
"""
from __future__ import annotations

import numpy as np
from scipy.special import lambertw

GAMMA_MAX = 40


# --------------------------------------------------------------------------- landmark formulas
def expected_tokens(alpha, gamma):
    """E[N] = (1 - alpha^(gamma+1)) / (1 - alpha); equals gamma+1 at alpha = 1."""
    alpha = np.asarray(alpha, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (1.0 - alpha ** (gamma + 1.0)) / (1.0 - alpha)
    return np.where(alpha >= 1.0, gamma + 1.0, out)


def speedup(alpha, gamma, c):
    """S(gamma) = E[N] / (gamma*c + 1)."""
    return expected_tokens(alpha, gamma) / (np.asarray(gamma, dtype=float) * c + 1.0)


def gamma_star(alpha, c):
    """Real-valued maximiser of S(gamma) via the Lambert-W closed form.

    L = -ln(alpha), y = -c W_{-1}(-exp(-1 - L(1/c - 1))), gamma* = (y - L - c) / (L c).
    Returns 0.0 when speculation cannot help (alpha <= c).
    """
    alpha = float(alpha)
    c = float(c)
    if alpha <= c:
        return 0.0
    L = -np.log(alpha)
    z = -np.exp(-1.0 - L * (1.0 / c - 1.0))
    y = -c * lambertw(z, -1).real
    return float((y - L - c) / (L * c))


def best_integer_gamma(alpha, c, gamma_max=GAMMA_MAX):
    """Integer gamma in [0, gamma_max] maximising the landmark speedup, and that speedup."""
    g = np.arange(gamma_max + 1)
    s = speedup(alpha, g, c)
    k = int(np.argmax(s))
    return k, float(s[k])


# --------------------------------------------------------------------------- variable alpha
def beta_moments(a, b, K):
    """E[alpha^k] for k = 0..K under alpha ~ Beta(a, b): prod_{j<k} (a+j)/(a+b+j)."""
    j = np.arange(K)
    return np.concatenate([[1.0], np.cumprod((a + j) / (a + b + j))])


def beta_grid(m, kappa, n=20001):
    """Cell-centred grid on (0,1) with exact Beta(m*kappa, (1-m)*kappa) mass per cell.

    Using CDF differences keeps the weights finite even when the density diverges at 0 or 1.
    """
    from scipy.stats import beta as beta_dist

    edges = np.linspace(0.0, 1.0, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = np.diff(beta_dist.cdf(edges, m * kappa, (1.0 - m) * kappa))
    return centres, w / w.sum()


def threshold_policy(alpha, lam, c, gamma_max=GAMMA_MAX):
    """gamma(alpha) = largest k <= gamma_max with alpha^k >= lam*c (k = 0 if none).

    Vectorised over alpha. alpha^0 = 1, so k >= 1 requires alpha >= lam*c; when lam*c > 1 the
    answer is 0 for every alpha < 1.
    """
    alpha = np.clip(np.asarray(alpha, dtype=float), 1e-300, 1.0)
    thr = lam * c
    if thr > 1.0:
        return np.where(alpha >= 1.0, gamma_max, 0).astype(int)
    with np.errstate(divide="ignore"):
        la = np.log(alpha)
        k = np.where(la < 0, np.floor(np.log(thr) / np.where(la < 0, la, -1.0) + 1e-12), gamma_max)
    return np.clip(k, 0, gamma_max).astype(int)


def fit_lambda_persistent(alpha_grid, weights, c, gamma_max=GAMMA_MAX, tol=1e-12, max_iter=10_000):
    """Fixed point lam = E[N(gamma(alpha))] / E[gamma(alpha) c + 1] for alpha persistent per round."""
    lam = 1.0
    for it in range(max_iter):
        g = threshold_policy(alpha_grid, lam, c, gamma_max)
        num = np.sum(weights * expected_tokens(alpha_grid, g))
        den = np.sum(weights * (g * c + 1.0))
        new = num / den
        if abs(new - lam) < tol:
            return float(new), it + 1
        lam = new
    return float(lam), max_iter


# --------------------------------------------------------------------------- replay simulator
def next_reject(accept):
    """nz[t] = first index p >= t with accept[p] == False (len(accept) if none)."""
    T = len(accept)
    nz = np.empty(T + 1, dtype=np.int64)
    nz[T] = T
    for t in range(T - 1, -1, -1):
        nz[t] = t if not accept[t] else nz[t + 1]
    return nz


def per_token_gamma(alpha_hat, lam, c, gamma_max=GAMMA_MAX):
    """For every start t: largest k with prod_{i<k} alpha_hat[t+i] >= lam*c, capped at gamma_max.

    This is the per-token form of the threshold rule ("keep drafting while a1*a2*...*ak >= lam c").
    """
    a = np.clip(np.asarray(alpha_hat, dtype=float), 1e-300, 1.0)
    T = len(a)
    cl = np.concatenate([[0.0], np.cumsum(np.log(a))])  # non-increasing
    thr = np.log(lam * c) if lam * c > 0 else -np.inf
    # largest k with cl[t+k] - cl[t] >= thr  <=>  -cl[t+k] <= thr' with thr' = -cl[t] - thr
    neg = -cl  # non-decreasing
    k = np.searchsorted(neg, neg[:T] - thr, side="right") - 1 - np.arange(T)
    return np.clip(k, 0, gamma_max).astype(np.int64)


def replay(accept, gammas, c):
    """Walk one trace. accept[p] says whether a draft at position p would be accepted
    (pre-drawn as u_p < beta_p; each position is drafted at most once, so this is exact).

    gammas: int array of per-start gamma (or a scalar for a fixed policy). A round starting at t
    drafts g = min(gamma(t), T - t - 1) tokens, accepts min(next_reject - t, g) and produces
    accepts + 1 tokens. Returns (tokens, cost, rounds); tokens always equals T.
    """
    T = len(accept)
    nz = next_reject(accept)
    if np.isscalar(gammas):
        gammas = np.full(T, int(gammas), dtype=np.int64)
    gl = gammas.tolist()
    nzl = nz.tolist()
    t = 0
    cost = 0.0
    rounds = 0
    while t < T:
        g = gl[t]
        if g > T - t - 1:
            g = T - t - 1
        acc = nzl[t] - t
        if acc > g:
            acc = g
        cost += g * c + 1.0
        t += acc + 1
        rounds += 1
    return T, cost, rounds


def replay_boundaries(accept, gammas):
    """Round start positions for one walk (used for figures)."""
    T = len(accept)
    nz = next_reject(accept)
    if np.isscalar(gammas):
        gammas = np.full(T, int(gammas), dtype=np.int64)
    t, starts = 0, []
    while t < T:
        g = min(int(gammas[t]), T - t - 1)
        starts.append(t)
        t += min(int(nz[t]) - t, g) + 1
    return starts


def replay_rounds(nz, gammas, c, max_rounds):
    """Walk a long stream (given its next-reject array nz) for max_rounds rounds.

    gammas is a per-position int array. Returns (tokens, cost).
    """
    T = len(nz) - 1
    gl = gammas.tolist() if not np.isscalar(gammas) else None
    gfix = int(gammas) if gl is None else 0
    nzl = nz.tolist()
    t = 0
    cost = 0.0
    for _ in range(max_rounds):
        g = gl[t] if gl is not None else gfix
        if g > T - t - 1:
            g = T - t - 1
        acc = nzl[t] - t
        if acc > g:
            acc = g
        cost += g * c + 1.0
        t += acc + 1
        if t >= T:
            raise RuntimeError("stream too short for requested rounds")
    return t, cost


def fit_lambda_fixed_point(throughput_fn, lam0=1.0, tol=1e-12, max_iter=200):
    """Iterate lam <- throughput_fn(lam) (Dinkelbach-style). If it cycles instead of converging,
    return the visited lam with the highest throughput. Returns (lam, iterations, converged)."""
    lam = lam0
    best = (-np.inf, lam0)
    seen = []
    for it in range(max_iter):
        thr = throughput_fn(lam)
        if thr > best[0]:
            best = (thr, lam)
        if abs(thr - lam) < tol:
            return float(thr), it + 1, True
        if any(abs(thr - s) < tol for s in seen):
            break
        seen.append(thr)
        lam = thr
    return float(best[1]), it + 1, False


def replay_many(accepts, gamma_fn, c):
    """Pooled throughput = total tokens / total cost over a list of traces.

    gamma_fn(i) returns the gamma array (or scalar) for trace i.
    """
    tok = 0
    cost = 0.0
    for i, acc in enumerate(accepts):
        n, k, _ = replay(acc, gamma_fn(i), c)
        tok += n
        cost += k
    return tok / cost
