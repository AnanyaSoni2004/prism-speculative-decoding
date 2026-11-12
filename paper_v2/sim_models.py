"""Exact computations for the synthetic models in the paper (Models A and C). Writes sim_numbers.json."""
import json, sys
import numpy as np
from scipy import stats
sys.path.insert(0, "../experiments/qwen")
import prop3_markov as P

out = {}
g = lambda a, G: sum(a ** k for k in range(G + 1))

# ---------------- Model A: one alpha ~ Beta per round (mean m, concentration kappa), independent rounds
m, c, CAP = 0.7, 0.05, 40
xs = (np.arange(200000) + 0.5) / 200000
A = []
for kappa in (None, 50, 20, 10, 5):
    if kappa is None:
        mom = lambda k: m ** k
        w = None
        sd = 0.0
    else:
        a, b = m * kappa, (1 - m) * kappa
        mom = lambda k, a=a, b=b: float(np.prod([(a + j) / (a + b + j) for j in range(k)]))
        w = stats.beta.pdf(xs, a, b); w /= w.sum()
        sd = float(np.sqrt(m * (1 - m) / (kappa + 1)))
    EN = lambda G: sum(mom(k) for k in range(G + 1))
    thr = [EN(G) / (G * c + 1) for G in range(CAP + 1)]
    Gb = int(np.argmax(thr))
    lm = g(m, Gb) / (Gb * c + 1)
    second = sd ** 2 * sum(k * (k - 1) / 2 * m ** (k - 2) for k in range(2, Gb + 1))
    if w is None:
        orc = thr[Gb]
    else:
        lam = thr[Gb]
        for _ in range(500):
            gam = np.minimum(np.floor(np.log(lam * c) / np.log(xs)), CAP).clip(0)
            N = (1 - xs ** (gam + 1)) / (1 - xs)
            new = float((w * N).sum() / (w * (gam * c + 1)).sum())
            if abs(new - lam) < 1e-13: break
            lam = new
        orc = lam
    A.append(dict(kappa=kappa, sd=sd, best_gamma=Gb, exact=thr[Gb], landmark=lm, err_at_gamma6_pct=100 * (g(m, 6) / EN(6) - 1),
                  second_order_at_gamma6_pct=100 * sd ** 2 * sum(k * (k - 1) / 2 * m ** (k - 2) for k in range(2, 7)) / EN(6),
                  landmark_error_pct=100 * (lm / thr[Gb] - 1),
                  second_order_pct=100 * second / EN(Gb), oracle=orc, oracle_gain_pct=100 * (orc / thr[Gb] - 1)))
out["A"] = A

# ---------------- Model C: two Markov regimes (0.95 / 0.40), switch prob s per token
C = []
for c in (0.05, 0.2):
    for s in (0.01, 0.05, 0.2, 0.5):
        fx, best, th = P.part_a(s, c)
        G = fx[1]
        # share of rounds that start in the hard regime under the best fixed gamma
        Pm = []
        for z in (0, 1):
            _, _, outs = P.round_stats(np.eye(2)[z], G, s)
            Pm.append(sum(nb for _, nb in outs))
        Pm = np.array(Pm); pi = np.array([Pm[1, 0], Pm[0, 1]]); pi /= pi.sum()
        lm = g(0.675, G) / (G * c + 1)
        hm = 1 / (0.5 / g(0.95, G) + 0.5 / g(0.40, G)) / (G * c + 1)
        C.append(dict(c=c, s=s, best_gamma=G, fixed=fx[0], landmark=lm, harmonic=hm,
                      landmark_error_pct=100 * (lm / fx[0] - 1), hard_start_pct=100 * pi[1],
                      threshold=th[0], thr_g=[th[1], th[2]], optimum=best[0], opt_g=[best[1], best[2]],
                      thr_loss_pct=100 * (best[0] / th[0] - 1),
                      thr_gain_pct=100 * (th[0] / fx[0] - 1), opt_gain_pct=100 * (best[0] / fx[0] - 1)))
out["C"] = C
json.dump(out, open("sim_numbers.json", "w"), indent=1)
for r in A: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
for r in C: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
