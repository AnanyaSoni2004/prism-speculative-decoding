"""E4: follow-up run. Appends a "Follow-up" section to results/REPORT.md.

1. Model C: naive adaptive rule (A_k = alpha_state^k) and share of rounds starting in the hard state
2. Harmonic-mean prediction for Model C, plus a numeric check that 1/g is convex
3. Re-timing of target verify passes, one target loaded at a time (timing_verify.csv)
4. E3 replay under wall-clock cost T = gamma*c + v(gamma+1) (table3_gpt2_wallclock)
5. Model D with posterior-mean shrinkage of the noisy alpha estimate
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm

from prism.core import (
    GAMMA_MAX,
    best_integer_gamma,
    beta_grid,
    expected_tokens,
    fit_lambda_fixed_point,
    next_reject,
    replay_rounds,
    speedup,
    threshold_policy,
)
from prism.util import RESULTS, SEEDS, Timer, log_seeds, write_md_table

R = RESULTS
OUT = []  # markdown lines for the Follow-up section
P = OUT.append
FIT, EVAL = SEEDS["e2_fit"], SEEDS["e2_eval"]
A_EASY, A_HARD, C_E2 = 0.95, 0.4, 0.05
N_ROUNDS_C = 20_000


def verdict(ok):
    return "**matches**" if ok else "**does not match**"


# =========================================================================== 1. Model C extras
def c_stream(s, seed):
    """Identical RNG sequence to e2_variable_alpha.model_c, so results line up with e2_modelC.csv."""
    L = N_ROUNDS_C * (GAMMA_MAX + 2) + 10
    rng = np.random.default_rng(seed)
    flips = rng.uniform(size=L) < s
    state0 = rng.uniform() < 0.5
    easy = np.logical_xor(state0, np.cumsum(flips) % 2 == 1)
    alpha = np.where(easy, A_EASY, A_HARD)
    acc = rng.uniform(size=L) < alpha
    return easy, alpha, next_reject(acc)


def c_thr(nz, gam, c=C_E2):
    tok, cost = replay_rounds(nz, gam, c, N_ROUNDS_C)
    return tok / cost


def hard_start_share(nz, easy, g):
    nzl, el = nz.tolist(), easy.tolist()
    T = len(nzl) - 1
    t, hard = 0, 0
    for _ in range(N_ROUNDS_C):
        hard += not el[t]
        gg = min(g, T - t - 1)
        t += min(nzl[t] - t, gg) + 1
    return hard / N_ROUNDS_C


def item1():
    rows = []
    for s in [0.01, 0.05, 0.2]:
        easy_f, alpha_f, nz_f = c_stream(s, FIT)
        gB = int(np.argmax([c_thr(nz_f, g) for g in range(GAMMA_MAX + 1)]))
        lam, it, conv = fit_lambda_fixed_point(lambda l: c_thr(nz_f, threshold_policy(alpha_f, l, C_E2)))
        bf, nv, hs = [], [], []
        for sd in EVAL:
            easy, alpha, nz = c_stream(s, sd)
            bf.append(c_thr(nz, gB))
            nv.append(c_thr(nz, threshold_policy(alpha, lam, C_E2)))
            hs.append(hard_start_share(nz, easy, gB))
        rows.append(dict(switch_prob=s, best_fixed_gamma=gB, best_fixed_mean=np.mean(bf), best_fixed_sd=np.std(bf, ddof=1),
                         naive_lambda=lam, naive_converged=conv,
                         naive_gamma_easy=int(threshold_policy(np.array([A_EASY]), lam, C_E2)[0]),
                         naive_gamma_hard=int(threshold_policy(np.array([A_HARD]), lam, C_E2)[0]),
                         naive_mean=np.mean(nv), naive_sd=np.std(nv, ddof=1),
                         naive_gain_pct=100 * (np.mean(nv) / np.mean(bf) - 1),
                         hard_start_share_mean=np.mean(hs), hard_start_share_sd=np.std(hs, ddof=1)))
    d = pd.DataFrame(rows)
    e2c = pd.read_csv(R / "e2_modelC.csv")
    d["best_fixed_matches_e2"] = np.isclose(d.best_fixed_mean.values, e2c.best_fixed_mean.values, rtol=1e-12)
    d.to_csv(R / "e4_modelC_extra.csv", index=False)

    exp_gain = {0.01: 12, 0.05: 2, 0.2: -18}
    exp_hard = {0.01: 77, 0.05: 75, 0.2: 69}
    P("### F1. Model C: naive adaptive rule and hard-state round starts\n")
    P("Naive rule: γ = largest k ≤ 40 with α_state^k ≥ λc, using the state at round start as if it lasted the whole round; "
      "λ fitted by fixed-point iteration on the fit seed; evaluated on the same 5 seeds as E2 (best fixed γ reproduces "
      f"`e2_modelC.csv` exactly: {bool(d.best_fixed_matches_e2.all())}).\n")
    P("| s | best fixed (γ) | naive adaptive | naive γ easy / hard | naive gain | expected | share of rounds starting hard (best fixed γ) | expected |")
    P("|---|---|---|---|---|---|---|---|")
    ok1 = ok2 = True
    for _, r in d.iterrows():
        g_ok = abs(r.naive_gain_pct - exp_gain[r.switch_prob]) <= 2.0
        h_ok = abs(100 * r.hard_start_share_mean - exp_hard[r.switch_prob]) <= 2.0
        ok1 &= g_ok
        ok2 &= h_ok
        P(f"| {r.switch_prob:g} | {r.best_fixed_mean:.3f} ± {r.best_fixed_sd:.3f} ({r.best_fixed_gamma}) | "
          f"{r.naive_mean:.3f} ± {r.naive_sd:.3f} | {r.naive_gamma_easy} / {r.naive_gamma_hard} | {r.naive_gain_pct:+.1f}% | "
          f"≈{exp_gain[r.switch_prob]:+d}% | {100 * r.hard_start_share_mean:.1f}% ± {100 * r.hard_start_share_sd:.1f} | ≈{exp_hard[r.switch_prob]}% |")
    P(f"\n- (a) Naive-rule gain {verdict(ok1)} the expected values (tolerance ±2 percentage points).")
    P(f"- (b) Hard-start share {verdict(ok2)} the expected values (tolerance ±2 percentage points). The stationary share of "
      f"*positions* in the hard state is 50%, so rounds over-sample the hard regime.")
    capped = d[(d.naive_gamma_easy == GAMMA_MAX) | (d.naive_gamma_hard == GAMMA_MAX)]
    P(f"- Cap flag: the naive rule's γ hits the cap of 40 at s = {', '.join(f'{s:g}' for s in capped.switch_prob)}"
      f" (0.95^k ≥ λc holds up to k ≈ {np.log(d.naive_lambda.iloc[0] * C_E2) / np.log(A_EASY):.0f} at s = {d.switch_prob.iloc[0]:g}). "
      f"So in the easy state the naive rule's γ is set by the cap, not by the rule.\n" if len(capped) else
      "- Cap flag: the naive rule never reaches γ = 40.\n")
    return d


# =========================================================================== 2. harmonic mean
def g_poly(gamma):
    return np.polynomial.Polynomial(np.ones(gamma + 1))


def item2():
    e2c = pd.read_csv(R / "e2_modelC.csv")
    m = 0.5 * (A_EASY + A_HARD)
    kL, sL = best_integer_gamma(m, C_E2)
    g = g_poly(kL)
    H = 1.0 / (0.5 / g(A_EASY) + 0.5 / g(A_HARD)) / (kL * C_E2 + 1)
    rows = []
    for _, r in e2c.iterrows():
        rows.append(dict(switch_prob=r.switch_prob, gamma=kL, landmark=sL, harmonic_H=H,
                         simulated_fixed_mean=r.fixed_at_landmark_mean, simulated_fixed_sd=r.fixed_at_landmark_sd,
                         H_vs_landmark_pct=100 * (H / sL - 1),
                         sim_vs_landmark_pct=100 * (r.fixed_at_landmark_mean / sL - 1),
                         sim_vs_H_pct=100 * (r.fixed_at_landmark_mean / H - 1)))
    d = pd.DataFrame(rows)
    d.to_csv(R / "e4_harmonic.csv", index=False)

    x = np.linspace(0.0, 1.0, 100_001)
    worst = (np.inf, None, None)
    for gam in range(1, 61):
        gp = g_poly(gam)
        d1, d2 = gp.deriv(1), gp.deriv(2)
        val = 2 * d1(x) ** 2 - gp(x) * d2(x)
        i = int(np.argmin(val))
        if val[i] < worst[0]:
            worst = (float(val[i]), gam, float(x[i]))
    # exact companion check: all integer coefficients of 2g'^2 - g g'' non-negative => >= 0 on [0, 1]
    coef_ok = True
    for gam in range(1, 61):
        gp = np.polynomial.Polynomial(np.ones(gam + 1, dtype=object))
        q = 2 * gp.deriv(1) ** 2 - gp * gp.deriv(2)
        coef_ok &= all(int(cf) >= 0 for cf in q.coef)
    conv = dict(min_value=worst[0], at_gamma=worst[1], at_alpha=worst[2], passes=worst[0] >= -1e-12,
                all_integer_coefficients_nonnegative=coef_ok)
    (R / "e4_convexity_check.json").write_text(json.dumps(conv, indent=2))

    P("### F2. Harmonic-mean prediction for Model C\n")
    P(f"At the landmark γ = {kL} (stationary mean α = {m}), H = (½/g(0.95) + ½/g(0.4))⁻¹ ÷ (γc + 1) with g(α) = Σ_(k=0..γ) α^k.\n")
    P("| s | landmark | H | simulated fixed-γ | H vs landmark | simulated vs landmark | simulated vs H |")
    P("|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        P(f"| {r.switch_prob:g} | {r.landmark:.3f} | {r.harmonic_H:.3f} | {r.simulated_fixed_mean:.3f} ± {r.simulated_fixed_sd:.3f} | "
          f"{r.H_vs_landmark_pct:+.1f}% | {r.sim_vs_landmark_pct:+.1f}% | {r.sim_vs_H_pct:+.1f}% |")
    hv = d.H_vs_landmark_pct.iloc[0]
    P(f"\n- H is {-hv:.1f}% below the landmark at γ = {kL}: {verdict(kL == 5 and abs(hv + 9.3) <= 0.3)} the expected ≈9.3% at γ = 5. "
      f"H is the s → 0 limit (each regime long enough to act as its own stream). As s grows, the simulation moves from H "
      f"toward the landmark.")
    P(f"- Convexity check 2g′² − g·g″ ≥ −1e-12 on [0, 1] (100,001 points, γ = 1…60): minimum {conv['min_value']:.3g} at "
      f"γ = {conv['at_gamma']}, α = {conv['at_alpha']:.4f} → **{'passes' if conv['passes'] else 'fails'}**. As an exact companion check, "
      f"every integer coefficient of 2g′² − g·g″ is non-negative for γ = 1…60: {coef_ok}. Because "
      f"(1/g)″ = (2g′² − g·g″)/g³, 1/g is convex, so by Jensen the harmonic-type average never exceeds the landmark.\n")
    return d


# =========================================================================== 3. re-timing
def item3(skip_models):
    path = R / "timing_verify.csv"
    if skip_models:
        path.unlink(missing_ok=True)
        P("### F3. Re-timing of verify passes\n\n_Skipped (`--skip-models`)._\n")
        return None
    import psutil
    import torch
    from transformers import AutoModelForCausalLM

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()

    prompt = np.load(R / "traces_large_prose.npz")["token_ids"][0][:32].tolist()
    rows = []
    for pair, name in [("large", "gpt2-large"), ("medium", "gpt2-medium")]:
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()
        draft = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float32).to(dev).eval()
        target = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
        vm0, sw0 = psutil.virtual_memory(), psutil.swap_memory()
        times = {}
        with torch.no_grad():
            for m in range(1, 10):
                ts = []
                for r in range(5 + 20):
                    out = target(torch.tensor([prompt], device=dev), use_cache=True)
                    new = torch.tensor([prompt[:m]], device=dev)
                    sync()
                    t0 = time.perf_counter()
                    target(new, past_key_values=out.past_key_values, use_cache=True)
                    sync()
                    if r >= 5:
                        ts.append(time.perf_counter() - t0)
                times[m] = np.array(ts)
        vm1, sw1 = psutil.virtual_memory(), psutil.swap_memory()
        for m in range(1, 10):
            rows.append(dict(pair=pair, model=name, device=str(dev), m=m, mean_s=times[m].mean(), sd_s=times[m].std(ddof=1),
                             median_s=float(np.median(times[m])),
                             ratio=1.0 if m == 1 else times[m].mean() / times[1].mean(),
                             ratio_median=1.0 if m == 1 else float(np.median(times[m]) / np.median(times[1])),
                             avail_mem_gb_before=vm0.available / 2**30, avail_mem_gb_after=vm1.available / 2**30,
                             swap_used_gb_before=sw0.used / 2**30, swap_used_gb_after=sw1.used / 2**30))
        del target, draft
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()
    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)

    old = pd.read_csv(R / "timing.csv").dropna(subset=["c"]).set_index("pair")
    P("### F3. Re-timing of verify passes (one target and the draft loaded at a time)\n")
    P("One cached target pass over m new tokens after a 32-token prefix, 20 repeats after 5 warm-ups. v(m) = mean time ÷ mean "
      "time at m = 1.\n")
    piv = d.pivot(index="m", columns="pair", values="ratio")
    med = d.pivot(index="m", columns="pair", values="ratio_median")
    ms = d.pivot(index="m", columns="pair", values="mean_s")
    pairs = [p for p in ["large", "medium"] if p in piv]
    P("| m | " + " | ".join(f"{p}: v(m) mean (median) | {p}: ms" for p in pairs) + " |")
    P("|---|" + "---|---|" * len(pairs))
    for m in piv.index:
        P(f"| {m} | " + " | ".join(f"{piv.loc[m, p]:.2f} ({med.loc[m, p]:.2f}) | {1000 * ms.loc[m, p]:.1f}" for p in pairs) + " |")
    for p in pairs:
        r = d[d.pair == p].iloc[0]
        P(f"\n- {p}: available RAM {r.avail_mem_gb_before:.2f} GB before / {r.avail_mem_gb_after:.2f} GB after timing; swap used "
          f"{r.swap_used_gb_before:.2f} → {r.swap_used_gb_after:.2f} GB. 8-token ratio from E3 in this same run "
          f"(both targets loaded in turn, `timing.csv`) {old.loc[p, 'verify_ratio_8_over_1']:.2f}×; isolated now {piv.loc[8, p]:.2f}×.")
    new8 = piv.loc[8, "large"] if "large" in piv else np.nan
    P(f"- The earlier run's 4.55× (gpt2-large, m = 8) is not in these results files any more; neither this run's E3 timing "
      f"({old.loc['large', 'verify_ratio_8_over_1']:.2f}×) nor the isolated timing ({new8:.2f}×) reproduces it, which fits a "
      f"transient effect such as memory pressure but does not prove it. Even in isolation only "
      f"{d[d.pair == 'large'].avail_mem_gb_before.iloc[0]:.1f} GB RAM was free with gpt2-large loaded, and v(m) for gpt2-large is "
      f"not monotone in m (a non-monotone v means noisy timing, not a real cost structure).\n")
    return d


# =========================================================================== 4. wall-clock replay
G_WC = 8  # v(m) is measured for m <= 9, so gamma <= 8


def replay_cost(accept, gammas, cost_tab):
    T = len(accept)
    nz = next_reject(accept).tolist()
    gl = [gammas] * T if np.isscalar(gammas) else gammas.tolist()
    t, cost = 0, 0.0
    while t < T:
        g = min(int(gl[t]), T - t - 1, G_WC)
        acc = min(nz[t] - t, g)
        cost += cost_tab[g]
        t += acc + 1
    return T, cost


def thr_cost(betas, U, gfn, cost_tab):
    out = []
    for u in U:
        tok = cost = 0.0
        for i, b in enumerate(betas):
            n, k = replay_cost(u[i] < b, gfn(i), cost_tab)
            tok += n
            cost += k
        out.append(tok / cost)
    return out


def marginal_gamma(ahat, lam, delta):
    """Keep drafting draft k while prod_{i<=k} ahat >= lam * delta_k (delta_k = marginal cost of draft k), cap G_WC."""
    a = np.clip(np.asarray(ahat, float), 1e-300, 1.0)
    T = len(a)
    la = np.concatenate([np.log(a), np.full(G_WC, -np.inf)])
    win = np.lib.stride_tricks.sliding_window_view(la, G_WC)[:T]
    P_ = np.exp(np.cumsum(win, axis=1))
    ok = P_ >= lam * delta[None, :]
    return np.where(ok.all(1), G_WC, np.argmin(ok, axis=1)).astype(np.int64)


def item4(vt, skip_models):
    path = R / "table3_gpt2_wallclock.csv"
    if skip_models or vt is None:
        for f in [path, R / "table3_gpt2_wallclock.md", R / "e4_wallclock_replay.csv"]:
            f.unlink(missing_ok=True)
        P("### F4. Wall-clock replay\n\n_Skipped (`--skip-models`)._\n")
        return None
    from e3_gpt2_traces import fit_calibrator

    timing = pd.read_csv(R / "timing.csv").dropna(subset=["c"]).set_index("pair")
    rows, cal_rows = [], []
    for pair in timing.index:
        c = float(timing.loc[pair, "c"])
        v = vt[vt.pair == pair].set_index("m")["ratio"].to_dict()
        v[1] = 1.0
        cost_tab = np.array([g * c + v[g + 1] for g in range(G_WC + 1)])
        delta = np.diff(cost_tab)  # marginal cost of draft k = c + v(k+1) - v(k), k = 1..G_WC
        for typ in ["prose", "maths", "code"]:
            z = np.load(R / f"traces_{pair}_{typ}.npz")
            beta, top, ent = z["beta"], z["draft_top"], z["draft_entropy"]
            n = len(beta)
            perm = np.random.default_rng(SEEDS["e3_split"]).permutation(n)
            tr, te = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
            U = [np.random.default_rng(s).uniform(size=beta.shape) for s in SEEDS["e3_replay"]]
            Utr, Ute = [u[tr] for u in U], [u[te] for u in U]
            btr, bte = beta[tr], beta[te]
            res = {}
            res["no_spec"] = ("0", thr_cost(bte, Ute, lambda i: 0, cost_tab))
            res["fixed_4"] = ("4", thr_cost(bte, Ute, lambda i: 4, cost_tab))
            kL, _ = best_integer_gamma(float(btr.mean()), c)
            kL = min(kL, G_WC)
            res["landmark"] = (str(kL), thr_cost(bte, Ute, lambda i: kL, cost_tab))
            trf = [np.mean(thr_cost(btr, Utr, lambda i, g=g: g, cost_tab)) for g in range(G_WC + 1)]
            gB = int(np.argmax(trf))
            res["best_fixed"] = (str(gB), thr_cost(bte, Ute, lambda i: gB, cost_tab))

            def ad(bsub, hsub, Us, lam):
                gam = [marginal_gamma(h, lam, delta) for h in hsub]
                return np.mean(thr_cost(bsub, Us, lambda i: gam[i], cost_tab))

            lo, _, conv_o = fit_lambda_fixed_point(lambda l: ad(btr, btr, Utr, l))
            cal_rows.append(dict(pair=pair, type=typ, calibrator="oracle (true beta)", train_throughput=ad(btr, btr, Utr, lo),
                                 lam=lo, converged=conv_o))
            gte = [marginal_gamma(h, lo, delta) for h in bte]
            res["adaptive_oracle"] = (f"lam={lo:.3f}", thr_cost(bte, Ute, lambda i: gte[i], cost_tab))
            cal = fit_calibrator(top[tr].ravel(), ent[tr].ravel(), btr.ravel())
            best = None
            for name, fn in cal.items():
                htr = fn(top[tr].ravel(), ent[tr].ravel()).reshape(btr.shape)
                lr, _, conv_r = fit_lambda_fixed_point(lambda l: ad(btr, htr, Utr, l))
                val = ad(btr, htr, Utr, lr)
                cal_rows.append(dict(pair=pair, type=typ, calibrator=name, train_throughput=val, lam=lr, converged=conv_r))
                if best is None or val > best[0]:
                    best = (val, name, lr)
            _, cname, lr = best
            hte = cal[cname](top[te].ravel(), ent[te].ravel()).reshape(bte.shape)
            gre = [marginal_gamma(h, lr, delta) for h in hte]
            res["adaptive_realistic"] = (f"{cname}, lam={lr:.3f}", thr_cost(bte, Ute, lambda i: gre[i], cost_tab))
            bf, orc, rea = (np.mean(res[k][1]) for k in ("best_fixed", "adaptive_oracle", "adaptive_realistic"))
            for pol, (g, vals) in res.items():
                rows.append(dict(pair=pair, type=typ, c=c, policy=pol, gamma=g, thr_mean=float(np.mean(vals)),
                                 thr_sd=float(np.std(vals, ddof=1)),
                                 recovery_pct=float(100 * (rea - bf) / (orc - bf)) if pol == "adaptive_realistic" and orc != bf else np.nan,
                                 best_fixed_at_cap=gB == G_WC))
    d = pd.DataFrame(rows)
    d.to_csv(R / "e4_wallclock_replay.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(R / "e4_wallclock_calibrators.csv", index=False)

    d["cell"] = d.apply(lambda r: f"{r.thr_mean:.3f} ± {r.thr_sd:.3f}", axis=1)
    order = ["no_spec", "fixed_4", "landmark", "best_fixed", "adaptive_oracle", "adaptive_realistic"]
    p = d.pivot_table(index=["pair", "type", "c"], columns="policy", values="cell", aggfunc="first")[order]
    gg = d.pivot_table(index=["pair", "type", "c"], columns="policy", values="gamma", aggfunc="first")
    p["landmark_gamma"], p["best_fixed_gamma"], p["realistic_calibrator"] = gg["landmark"], gg["best_fixed"], gg["adaptive_realistic"]
    p["realistic_recovery_pct"] = d[d.policy == "adaptive_realistic"].set_index(["pair", "type", "c"])["recovery_pct"].round(1)
    p = p.reset_index()
    p["c"] = p["c"].round(4)
    p.to_csv(path, index=False)
    write_md_table(p, R / "table3_gpt2_wallclock.md")

    old = pd.read_csv(R / "e3_replay.csv").query("c_label == 'measured'")
    P("### F4. Wall-clock replay (Table 3, second version)\n")
    P("Round cost T = γc + v(γ + 1), with c from `timing.csv` and v(m) from F3 (v(1) = 1). γ is capped at 8 because v is measured "
      "only up to m = 9. Best fixed γ, λ and calibrators are re-selected on the training half under this cost. The adaptive rule "
      "drafts the k-th token while ∏β ≥ λ·Δ_k, where Δ_k = c + v(k+1) − v(k) is that token's marginal cost. Test half, mean ± SD "
      "over 5 replay seeds, in units of one cached target step.\n")
    for pair in timing.index:
        c = float(timing.loc[pair, "c"])
        P(f"- {pair}: c = {c:.3f}; T(γ) for γ = 0…8: " +
          ", ".join(f"{g * c + (1.0 if g == 0 else vt[(vt.pair == pair) & (vt.m == g + 1)].ratio.iloc[0]):.2f}" for g in range(G_WC + 1)))
    P("")
    P((R / "table3_gpt2_wallclock.md").read_text())
    slower = d[(d.policy != "no_spec") & (d.thr_mean < 1.0)]
    bf_slow = d[(d.policy == "best_fixed") & (d.thr_mean < 1.0)]
    P("Change from the linear-cost Table 3 (best fixed γ and adaptive oracle, test half):\n")
    P("| pair | type | best fixed: linear → wall-clock | adaptive oracle: linear → wall-clock |\n|---|---|---|---|")
    for (pair, typ), g in d.groupby(["pair", "type"], sort=False):
        o = old[(old.pair == pair) & (old.type == typ)].set_index("policy")
        n_ = g.set_index("policy")
        P(f"| {pair} | {typ} | {o.loc['best_fixed', 'thr_mean']:.3f} (γ={o.loc['best_fixed', 'gamma']}) → "
          f"{n_.loc['best_fixed', 'thr_mean']:.3f} (γ={n_.loc['best_fixed', 'gamma']}) | "
          f"{o.loc['adaptive_oracle', 'thr_mean']:.3f} → {n_.loc['adaptive_oracle', 'thr_mean']:.3f} |")
    P(f"\n- Slower than plain decoding (throughput < 1) under wall-clock cost: "
      + ("none of the policies in any pair/type cell." if slower.empty else
         "; ".join(f"{r.pair}/{r.type} {r.policy} {r.thr_mean:.3f}" for _, r in slower.iterrows()) + ".")
      + (" The re-selected best fixed γ is never slower than plain decoding" if bf_slow.empty else
         f" The best fixed γ itself is below 1 in {len(bf_slow)} cell(s)")
      + (f"; it is γ = 0 (plain decoding, no speedup) in: "
         + ", ".join(f"{r.pair}/{r.type}" for _, r in d[(d.policy == 'best_fixed') & (d.gamma == '0')].iterrows()) + "."
         if ((d.policy == "best_fixed") & (d.gamma == "0")).any() else "."))
    cap = d[(d.policy == "best_fixed") & d.best_fixed_at_cap]
    P(f"- Best fixed γ hits the cap of 8 in {cap[['pair', 'type']].drop_duplicates().shape[0]} cell(s).")
    cr = pd.DataFrame(cal_rows)
    nc = cr[~cr.converged]
    P(f"- λ fixed-point fits that cycled instead of converging (best visited λ kept): {len(nc)} of {len(cr)}"
      + (" — " + "; ".join(f"{r.pair}/{r.type} {r.calibrator} (λ = {r.lam:.3f})" for _, r in nc.iterrows()) if len(nc) else "") + ".")
    neg = d[(d.policy == "adaptive_realistic") & (d.recovery_pct < 0)]
    if len(neg):
        P("- The realistic policy is *below* the best fixed γ in " + ", ".join(f"{r.pair}/{r.type} (recovery {r.recovery_pct:.0f}%)"
          for _, r in neg.iterrows()) + ". *Guess:* with a non-monotone v(m), Δ_k is negative for some k, so the "
          "per-token rule becomes erratic and the fixed-point fit can stall; the oracle's gain in that cell is also small.")
    P("")
    return d


# =========================================================================== 5. Model D shrinkage
def item5():
    m, kappa, c = 0.7, 10, 0.05
    a_, b_ = m * kappa, (1 - m) * kappa
    A = pd.read_csv(R / "e2_modelA_main.csv")
    ref = A[A.kappa == 10].iloc[0]
    bf, orc = ref.best_fixed_true, ref.adaptive_oracle
    xg, wg = beta_grid(m, kappa, 20001)
    D_old = pd.read_csv(R / "e2_modelD.csv").set_index("sigma")

    def post_mean_fn(sigma):
        if sigma == 0:
            return lambda ah: ah
        grid = np.linspace(0.01, 0.99, 1961)
        like = norm.pdf((grid[:, None] - xg[None, :]) / sigma)
        pm = (like * wg * xg).sum(1) / (like * wg).sum(1)
        lo_l, hi_l = norm.cdf((0.01 - xg) / sigma), norm.sf((0.99 - xg) / sigma)  # clipped endpoints are censored
        pm_lo = np.sum(lo_l * wg * xg) / np.sum(lo_l * wg)
        pm_hi = np.sum(hi_l * wg * xg) / np.sum(hi_l * wg)

        def f(ah):
            out = np.interp(ah, grid, pm)
            out = np.where(ah <= 0.01, pm_lo, out)
            return np.where(ah >= 0.99, pm_hi, out)
        return f

    def draw(seed, sigma, n=200_000):  # same RNG sequence as e2_variable_alpha.model_d
        rng = np.random.default_rng(seed)
        al = rng.beta(a_, b_, size=n)
        ah = np.clip(al + rng.normal(0.0, sigma, size=n), 0.01, 0.99) if sigma > 0 else al
        return al, ah

    def thr(al, est, lam):
        g = threshold_policy(est, lam, c)
        return np.sum(expected_tokens(al, g)) / np.sum(g * c + 1.0)

    rows = []
    for sigma in [0.0, 0.05, 0.1, 0.2]:
        f = post_mean_fn(sigma)
        al_f, ah_f = draw(FIT, sigma)
        out = dict(sigma=sigma)
        for name, est in [("raw", lambda ah: ah), ("shrunk", f)]:
            e_f = est(ah_f)
            lam, _, conv = fit_lambda_fixed_point(lambda l: thr(al_f, e_f, l))
            vals = [thr(al, est(ah), lam) for al, ah in (draw(s, sigma) for s in EVAL)]
            out[f"{name}_mean"], out[f"{name}_sd"], out[f"{name}_lambda"] = np.mean(vals), np.std(vals, ddof=1), lam
            out[f"{name}_survival_pct"] = 100 * (np.mean(vals) - bf) / (orc - bf)
        out["raw_matches_e2"] = bool(np.isclose(out["raw_mean"], D_old.loc[sigma, "throughput_mean"], rtol=1e-12))
        out["best_fixed_true"], out["adaptive_oracle_exact"] = bf, orc
        rows.append(out)
    d = pd.DataFrame(rows)
    d.to_csv(R / "e4_modelD_shrinkage.csv", index=False)

    P("### F5. Model D with posterior-mean shrinkage\n")
    P("The policy uses E[α | α̂] under the known Beta(7, 3) prior, computed on the 20,001-point α grid. Gaussian likelihood "
      "for interior α̂; the clipped endpoints 0.01 and 0.99 are treated as censored. λ is refit per σ on the fit seed; 200k "
      f"rounds × 5 seeds. The raw-α̂ policy reproduces `e2_modelD.csv` exactly: {bool(d.raw_matches_e2.all())}.\n")
    P(f"| σ | raw α̂: throughput | raw: share of oracle gain kept | shrunk: throughput | shrunk: share of oracle gain kept |")
    P("|---|---|---|---|---|")
    for _, r in d.iterrows():
        P(f"| {r.sigma:g} | {r.raw_mean:.3f} ± {r.raw_sd:.3f} | {r.raw_survival_pct:.0f}% | {r.shrunk_mean:.3f} ± {r.shrunk_sd:.3f} | "
          f"{r.shrunk_survival_pct:.0f}% |")
    P(f"\nOracle {orc:.3f}, best fixed {bf:.3f} (exact, Model A, κ = 10). No expected values were given for this item.\n")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()
    log_seeds("e4", {"model_c_fit": FIT, "model_c_eval": EVAL, "e3_split": SEEDS["e3_split"], "e3_replay": SEEDS["e3_replay"]})
    t0 = time.time()
    P("## Follow-up\n")
    P("Produced by `e4_followup.py`, appended after the main report. Every number below is read from that run's output files "
      "(`e4_*.csv`, `e4_convexity_check.json`, `timing_verify.csv`, `table3_gpt2_wallclock.*`).\n")
    item1()
    item2()
    vt = item3(args.skip_models)
    item4(vt, args.skip_models)
    item5()
    P("### F6. Figures\n")
    P("No table that `figures.py` reads (e1_*, e2_modelA_main, e3_beta_stats, timing.csv, traces, e3_fig4_example) was changed "
      "by this follow-up; E4 writes only new files, so the figures were not regenerated.\n")
    P(f"E4 runtime: {time.time() - t0:.0f} s.\n")

    rep = R / "REPORT.md"
    text = rep.read_text() if rep.exists() else ""
    if "\n## Follow-up\n" in text:  # make re-running e4 idempotent
        text = text.split("\n## Follow-up\n")[0].rstrip() + "\n"
    rep.write_text(text + "\n" + "\n".join(OUT) + "\n")
    print("\n".join(OUT))


if __name__ == "__main__":
    with Timer("e4"):
        main()
