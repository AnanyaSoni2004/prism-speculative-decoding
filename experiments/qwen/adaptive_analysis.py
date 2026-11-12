"""Adaptive-length analysis on own-text traces (exact replay with common random numbers).

Usage: python adaptive_analysis.py prose=PATH maths=PATH code=PATH
Policies (all tuned on one half of the passages, tested on the other, 2-fold):
  fixed      best fixed gamma
  heuristic  Hugging Face style schedule: +2 after a fully accepted round, -1 otherwise
  per-token  stop after the first draft token whose predicted acceptance < tau   (DISCO-style)
  survival   stop after the predicted survival probability falls below theta, calibrated
             from draft confidence (max prob, entropy)                           (SpecDec++-style; our rule with estimated A_k)
             Both pay for every draft step they run, including the one that triggers the stop.
  oracle     the same survival rule with the true beta_t
  hindsight  drafts exactly up to the first rejection (unreachable upper bound)
"""
import sys
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CAP, R = 40, 40
# (c, v): draft cost and cost of verifying 2+ tokens, in target steps. v = 1 is assumption (A2);
# (0.66, 1.2) is what time_models.py measured for Qwen2.5-0.5B -> 3B on a T4 (see timing.json).
COSTS = [(0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.66, 1.2)]
rng0 = np.random.default_rng(0)


def run_policy(kind, par, acc, pred, c, v=1.0):
    """acc: (R, T) bool accept draws for one passage; pred: (T,) predicted acceptance.
    Returns (tokens, time) summed over the R replays."""
    Rn, T = acc.shape
    tokens = time = 0.0
    for r in range(Rn):
        a = acc[r]
        t, g_state = 0, 5
        while t < T - 1:
            room = min(CAP, T - 1 - t)
            if kind == "fixed":
                g = min(par, room)
            elif kind == "heuristic":
                g = min(g_state, room)
            elif kind == "pertoken":          # stop after the first low-confidence token (it is kept: already paid for)
                g = 0
                while g < room:
                    g += 1
                    if pred[t + g - 1] < par:
                        break
            elif kind == "survival_cal":      # same, on the running product of predicted acceptances
                g, s = 0, 1.0
                while g < room:
                    s *= pred[t + g]
                    g += 1
                    if s < par:
                        break
            elif kind == "survival":          # oracle form: knows beta before drafting, so no step is wasted
                g, s = 0, 1.0
                while g < room:
                    s *= pred[t + g]
                    if s < par:
                        break
                    g += 1
            elif kind == "hindsight":
                g = 0
                while g < room and a[t + g]:
                    g += 1
            k = 0
            while k < g and a[t + k]:
                k += 1
            if kind == "heuristic":
                g_state = min(CAP, g_state + 2) if (k == g and g > 0) else max(1, g_state - 1)
            tokens += k + 1
            time += g * c + (1.0 if g == 0 else v)
            t += k + 1
    return tokens, time


def evaluate(kind, par, idx, ACC, PRED, c):
    c, v = c
    res = np.array([run_policy(kind, par, ACC[i], PRED[i] if PRED is not None else None, c, v) for i in idx])
    return res  # (n, 2)


def tune(kind, grid, idx, ACC, PRED, c):
    best, bp = -1, None
    for par in grid:
        r = evaluate(kind, par, idx, ACC, PRED, c)
        thr = r[:, 0].sum() / r[:, 1].sum()
        if thr > best:
            best, bp = thr, par
    return bp


def ci(num, den, base_num, base_den, B=4000):
    """Throughput, and % gain over the baseline with a paired bootstrap 95% CI over passages."""
    n = len(num)
    ix = rng0.integers(0, n, (B, n))
    thr = num.sum() / den.sum()
    g = 100 * ((num[ix].sum(1) / den[ix].sum(1)) / (base_num[ix].sum(1) / base_den[ix].sum(1)) - 1)
    return thr, 100 * (thr / (base_num.sum() / base_den.sum()) - 1), np.percentile(g, 2.5), np.percentile(g, 97.5)


def analyse(name, path):
    z = np.load(path)
    beta, a, qmax, qent = (z[k].astype(np.float64) for k in ("own_beta", "own_a", "own_qmax", "own_qent"))
    n, T = beta.shape
    rng = np.random.default_rng(1)
    ACC = rng.random((n, R, T)) < a[:, None, :]          # exact replay draws, shared by every policy
    folds = [np.arange(0, n // 2), np.arange(n // 2, n)]
    print(f"\n===== {name}: {n} passages x {T} tokens, mean beta {beta.mean():.3f}")

    # like-for-like check against the end-to-end runs (first 30 passages, complete rounds only)
    for g in (3, 6):
        tok = rnd = 0
        per = []
        for i in range(30):
            tk = rd = 0
            for r in range(R):
                t = 0
                while t + g < T:
                    k = 0
                    while k < g and ACC[i, r, t + k]:
                        k += 1
                    tk += k + 1; rd += 1; t += k + 1
            per.append((tk, rd))
        per = np.array(per, float)
        ix = rng0.integers(0, 30, (4000, 30))
        bs = per[ix, 0].sum(1) / per[ix, 1].sum(1)
        print(f"  exact replay, first 30 passages, gamma={g}: {per[:,0].sum()/per[:,1].sum():.3f} "
              f"[{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")

    for c in COSTS:
        out = {k: [] for k in ("fixed", "heuristic", "pertoken", "survival", "oracle", "hindsight")}
        chosen = []
        for tr, te in (folds, folds[::-1]):
            # calibrators: draft confidence -> beta, fitted on the training half
            iso = IsotonicRegression(y_min=0.01, y_max=0.999, out_of_bounds="clip").fit(qmax[tr].ravel(), beta[tr].ravel())
            P_iso = iso.predict(qmax.ravel()).reshape(n, T)
            X = lambda ix: np.c_[qmax[ix].ravel(), qent[ix].ravel(), qmax[ix].ravel() ** 2]
            lr = LogisticRegression(C=10, max_iter=2000)
            # fit on soft targets by duplicating rows with weights beta / 1-beta
            Xt = X(tr); w = beta[tr].ravel()
            lr.fit(np.r_[Xt, Xt], np.r_[np.ones(len(w)), np.zeros(len(w))], sample_weight=np.r_[w, 1 - w])
            P_lr = np.clip(lr.predict_proba(X(np.arange(n)))[:, 1], 0.01, 0.999).reshape(n, T)
            P = P_lr if ((P_lr[tr] - beta[tr]) ** 2).mean() < ((P_iso[tr] - beta[tr]) ** 2).mean() else P_iso
            thetas = np.r_[0.02, 0.05, np.arange(0.1, 0.96, 0.05)]
            gf = tune("fixed", range(0, 21), tr, ACC, None, c)
            tau = tune("pertoken", thetas, tr, ACC, P, c)
            th = tune("survival_cal", thetas, tr, ACC, P, c)
            tho = tune("survival", thetas, tr, ACC, beta, c)
            chosen.append((gf, round(float(tau), 2), round(float(th), 2), round(float(tho), 2)))
            out["fixed"].append(evaluate("fixed", gf, te, ACC, None, c))
            out["heuristic"].append(evaluate("heuristic", None, te, ACC, None, c))
            out["pertoken"].append(evaluate("pertoken", tau, te, ACC, P, c))
            out["survival"].append(evaluate("survival_cal", th, te, ACC, P, c))
            out["oracle"].append(evaluate("survival", tho, te, ACC, beta, c))
            out["hindsight"].append(evaluate("hindsight", None, te, ACC, None, c))
            r2 = 1 - ((P[te] - beta[te]) ** 2).mean() / beta[te].var()
        out = {k: np.concatenate(v) for k, v in out.items()}
        b = out["fixed"]
        print(f"  c={c[0]}, v={c[1]}: best fixed gamma per fold {[x[0] for x in chosen]}, tau {[x[1] for x in chosen]}, "
              f"theta {[x[2] for x in chosen]}, oracle theta {[x[3] for x in chosen]}, calibrator R^2 (last fold) {r2:.2f}")
        og = None
        for k in ("fixed", "heuristic", "pertoken", "survival", "oracle", "hindsight"):
            thr, gain, lo, hi = ci(out[k][:, 0], out[k][:, 1], b[:, 0], b[:, 1])
            if k == "oracle":
                og = gain
            print(f"     {k:<10} speedup {thr:.3f}x   gain over best fixed {gain:+5.1f}% [{lo:+.1f},{hi:+.1f}]")
        sg = 100 * ((out['survival'][:, 0].sum() / out['survival'][:, 1].sum()) / (b[:, 0].sum() / b[:, 1].sum()) - 1)
        print(f"     share of oracle gain recovered by calibrated survival rule: {100 * sg / og:.0f}%")


for arg in sys.argv[1:]:
    nm, p = arg.split("=", 1)
    analyse(nm, p)
