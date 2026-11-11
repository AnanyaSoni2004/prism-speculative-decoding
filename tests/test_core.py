import numpy as np
import pytest

from prism.core import (
    beta_grid,
    beta_moments,
    best_integer_gamma,
    expected_tokens,
    fit_lambda_persistent,
    gamma_star,
    per_token_gamma,
    replay,
    speedup,
    threshold_policy,
)


def test_expected_tokens_basic():
    assert expected_tokens(0.5, 0) == pytest.approx(1.0)
    assert expected_tokens(0.5, 1) == pytest.approx(1.5)
    assert expected_tokens(0.8, 3) == pytest.approx(1 + 0.8 + 0.64 + 0.512)
    assert expected_tokens(1.0, 7) == pytest.approx(8.0)
    assert expected_tokens(0.0, 5) == pytest.approx(1.0)


def test_speedup():
    assert speedup(0.7, 0, 0.05) == pytest.approx(1.0)
    assert speedup(0.7, 6, 0.05) == pytest.approx((1 - 0.7**7) / (0.3 * 1.3))
    assert speedup(0.7, 6, 0.05) == pytest.approx(2.353, abs=5e-4)


def test_gamma_star_known_value():
    assert gamma_star(0.8, 0.05) == pytest.approx(7.8567, abs=1e-4)
    k, s = best_integer_gamma(0.8, 0.05)
    assert k == 8 and s == pytest.approx(3.092, abs=5e-4)
    assert gamma_star(0.1, 0.2) == 0.0


@pytest.mark.parametrize("alpha,c", [(0.6, 0.05), (0.9, 0.02), (0.8, 0.2)])
def test_gamma_star_matches_grid(alpha, c):
    g = np.arange(0, 80, 1e-3)
    assert gamma_star(alpha, c) == pytest.approx(g[np.argmax(speedup(alpha, g, c))], abs=2e-3)


def test_beta_moments():
    m = beta_moments(3.5, 1.5, 6)
    assert m[0] == 1.0 and m[1] == pytest.approx(0.7)
    assert m.sum() == pytest.approx(3.4894, abs=1e-4)
    x, w = beta_grid(0.7, 5.0)
    for k in range(7):
        assert np.sum(w * x**k) == pytest.approx(m[k], abs=1e-6)


def test_threshold_policy():
    a = np.array([0.0, 0.3, 0.7, 0.999999, 1.0])
    g = threshold_policy(a, 2.0, 0.05)  # threshold 0.1
    # 0.3^1 = 0.3 >= 0.1, 0.3^2 = 0.09 < 0.1 -> 1;  0.7^6 = 0.1176, 0.7^7 = 0.082 -> 6
    assert list(g) == [0, 1, 6, 40, 40]
    assert list(threshold_policy(np.array([0.5, 0.9]), 30.0, 0.05)) == [0, 0]  # lam*c > 1
    # brute-force agreement
    rng = np.random.default_rng(0)
    for _ in range(200):
        al, lam, c = rng.uniform(0.01, 0.99), rng.uniform(0.5, 5), rng.uniform(0.01, 0.4)
        ks = [k for k in range(41) if al**k >= lam * c]
        assert threshold_policy(np.array([al]), lam, c)[0] == (max(ks) if ks else 0)


def test_fit_lambda_constant_alpha_equals_landmark():
    lam, _ = fit_lambda_persistent(np.array([0.7]), np.array([1.0]), 0.05)
    k, s = best_integer_gamma(0.7, 0.05)
    assert lam == pytest.approx(s, rel=1e-9)


def test_per_token_gamma_matches_bruteforce():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.05, 1.0, 60)
    lam, c = 2.0, 0.05
    g = per_token_gamma(a, lam, c)
    for t in range(60):
        best = 0
        for k in range(1, 41):
            if t + k > 60:
                break
            if np.prod(a[t : t + k]) >= lam * c:
                best = k
        assert g[t] == best


def _slow_replay(accept, gammas, c):
    T, t, cost = len(accept), 0, 0.0
    while t < T:
        g = min(int(gammas[t]), T - t - 1)
        acc = 0
        while acc < g and accept[t + acc]:
            acc += 1
        cost += g * c + 1
        t += acc + 1
    return cost


def test_replay_matches_slow_and_counts_tokens():
    rng = np.random.default_rng(2)
    for _ in range(50):
        acc = rng.uniform(size=128) < rng.uniform(0.2, 0.95)
        gam = rng.integers(0, 12, size=128)
        T, cost, _ = replay(acc, gam, 0.1)
        assert T == 128
        assert cost == pytest.approx(_slow_replay(acc, gam, 0.1))
    T, cost, rounds = replay(np.ones(10, bool), 0, 0.1)
    assert cost == 10 and rounds == 10  # gamma = 0 -> no speculation


def test_replay_constant_alpha_matches_landmark():
    rng = np.random.default_rng(3)
    alpha, c, g = 0.75, 0.05, 5
    tok = cost = 0
    for _ in range(400):
        acc = rng.uniform(size=2000) < alpha
        n, k, _ = replay(acc, g, c)
        tok += n
        cost += k
    assert tok / cost == pytest.approx(float(speedup(alpha, g, c)), rel=0.01)
