"""E1: the landmark under constant alpha. Writes e1_grid.csv, e1_fig1_curves.csv, e1_fig2_plane.npz."""
import numpy as np
import pandas as pd

from prism.core import GAMMA_MAX, best_integer_gamma, gamma_star, speedup
from prism.util import RESULTS, Timer, log_seeds


def main():
    log_seeds("e1", "none (deterministic)")
    rows = []
    for c in [0.02, 0.05, 0.1, 0.2, 0.4]:
        for a in np.round(np.arange(0.05, 0.951, 0.01), 2):
            gs = gamma_star(a, c)
            k, s = best_integer_gamma(a, c)
            rows.append(dict(c=c, alpha=a, gamma_star_real=gs, gamma_best_int=k, best_speedup=s,
                             speedup_at_real=float(speedup(a, gs, c)), hits_cap=k == GAMMA_MAX))
    pd.DataFrame(rows).to_csv(RESULTS / "e1_grid.csv", index=False)

    # Figure 1 data
    rows = []
    for a in [0.5, 0.7, 0.8, 0.9]:
        gs = gamma_star(a, 0.05)
        k, s = best_integer_gamma(a, 0.05)
        for g in range(21):
            rows.append(dict(alpha=a, c=0.05, gamma=g, speedup=float(speedup(a, g, 0.05)),
                             gamma_star_real=gs, gamma_best_int=k, best_speedup=s))
    pd.DataFrame(rows).to_csv(RESULTS / "e1_fig1_curves.csv", index=False)

    # Figure 2 data: best speedup and best integer gamma over the (alpha, c) plane
    al = np.arange(1, 100) / 100.0
    cs = np.arange(1, 100) / 100.0
    g = np.arange(GAMMA_MAX + 1)
    best_s = np.empty((len(cs), len(al)))
    best_g = np.empty((len(cs), len(al)), dtype=int)
    for i, c in enumerate(cs):
        s = speedup(al[None, :], g[:, None], c)
        best_s[i] = s.max(0)
        best_g[i] = s.argmax(0)
    np.savez(RESULTS / "e1_fig2_plane.npz", alpha=al, c=cs, best_speedup=best_s, best_gamma=best_g)

    t = pd.read_csv(RESULTS / "e1_grid.csv")
    sub = t[(t.c == 0.05) & t.alpha.isin([0.5, 0.6, 0.7, 0.8, 0.9])]
    print(sub.to_string(index=False))


if __name__ == "__main__":
    with Timer("e1"):
        main()
