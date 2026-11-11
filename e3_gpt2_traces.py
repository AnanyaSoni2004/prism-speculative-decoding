"""E3: real acceptance traces on GPT-2 pairs (inference only).

1. teacher-forced traces of beta_t = sum_x min(p_t(x), q_t(x)) on prose / maths / code
2. timing of draft and target per token (c), and of an 8-token verification pass
3. descriptive statistics of beta_t (is alpha constant?)
4. replay simulator with six policies, train/test split
5. Jensen check on real data
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
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from prism.core import (
    GAMMA_MAX,
    best_integer_gamma,
    fit_lambda_fixed_point,
    gamma_star,
    per_token_gamma,
    replay,
    replay_boundaries,
    speedup,
)
from prism.util import RESULTS, SEEDS, Timer, log_seeds

PREFIX, GEN = 32, 128
TOTAL = PREFIX + GEN
DRAFT = "distilgpt2"
TARGETS = {"large": "gpt2-large", "medium": "gpt2-medium"}
TYPES = ["prose", "maths", "code"]
C_SENS = [0.05, 0.1, 0.2]
DEV_LOG = RESULTS / "e3_deviations.json"


def note_deviation(msg):
    data = json.loads(DEV_LOG.read_text()) if DEV_LOG.exists() else []
    data.append(msg)
    DEV_LOG.write_text(json.dumps(data, indent=2))
    print("DEVIATION:", msg)


def pick_device(force_cpu=False):
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------- data
def load_passages(tok, n_per_type, seed):
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    out, counts = {}, {}

    def take(texts, name, cap):
        ids = [tok(t)["input_ids"] for t in texts]
        ok = [x[:TOTAL] for x in ids if len(x) >= TOTAL]
        counts[name] = {"candidates": len(texts), "long_enough": len(ok)}
        order = rng.permutation(len(ok))[: min(cap, len(ok))]
        out[name] = np.array([ok[i] for i in sorted(order)], dtype=np.int64)
        counts[name]["used"] = len(out[name])

    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    paras = [t.strip() for t in wt["text"] if t.strip() and not t.strip().startswith("=")]
    take(paras, "prose", n_per_type)
    gs = load_dataset("openai/gsm8k", "main", split="test")
    take([q + "\n" + a for q, a in zip(gs["question"], gs["answer"])], "maths", n_per_type)
    he = load_dataset("openai/openai_humaneval", split="test")
    take([p + s for p, s in zip(he["prompt"], he["canonical_solution"])], "code", n_per_type)
    return out, counts


def load_model(name, dev):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    return m.to(dev).eval()


# --------------------------------------------------------------------------- traces
@torch.no_grad()
def probs_for(model, ids, dev, bs):
    """Softmax (temperature 1) at the positions predicting tokens PREFIX..TOTAL-1."""
    res = []
    for i in range(0, len(ids), bs):
        x = torch.tensor(ids[i : i + bs], device=dev)
        logits = model(x).logits[:, PREFIX - 1 : TOTAL - 1, :].float()
        res.append(torch.softmax(logits, dim=-1))
    return res


@torch.no_grad()
def traces_for_pair(draft, target, ids, dev, bs):
    beta, top, ent = [], [], []
    for i in range(0, len(ids), bs):
        x = torch.tensor(ids[i : i + bs], device=dev)
        q = torch.softmax(draft(x).logits[:, PREFIX - 1 : TOTAL - 1, :].float(), -1)
        p = torch.softmax(target(x).logits[:, PREFIX - 1 : TOTAL - 1, :].float(), -1)
        beta.append(torch.minimum(p, q).sum(-1).cpu().numpy())
        top.append(q.max(-1).values.cpu().numpy())
        ent.append((-(q * torch.log(q.clamp_min(1e-30))).sum(-1)).cpu().numpy())
        del p, q
    return np.concatenate(beta), np.concatenate(top), np.concatenate(ent)


# --------------------------------------------------------------------------- timing
@torch.no_grad()
def time_model(model, prompt_ids, dev, n_tokens=64, warm=2, reps=5):
    per_tok = []
    for r in range(warm + reps):
        x = torch.tensor([prompt_ids], device=dev)
        out = model(x, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
        sync(dev)
        times = []
        for _ in range(n_tokens):
            t0 = time.perf_counter()
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            sync(dev)
            times.append(time.perf_counter() - t0)
        if r >= warm:
            per_tok.append(np.mean(times))
    return float(np.mean(per_tok)), float(np.std(per_tok, ddof=1))


@torch.no_grad()
def time_verify(model, prompt_ids, dev, k=8, warm=2, reps=5):
    """Time one cached forward over k new tokens vs one cached forward over 1 token."""
    res = {}
    for n in (1, k):
        ts = []
        for r in range(warm + reps):
            out = model(torch.tensor([prompt_ids], device=dev), use_cache=True)
            new = torch.tensor([prompt_ids[:n]], device=dev)
            sync(dev)
            t0 = time.perf_counter()
            model(new, past_key_values=out.past_key_values, use_cache=True)
            sync(dev)
            if r >= warm:
                ts.append(time.perf_counter() - t0)
        res[n] = float(np.mean(ts))
    return res[1], res[k]


# --------------------------------------------------------------------------- replay
def throughput(betas, u_list, gamma_fn, c):
    """Pooled tokens/cost over passages for one seed's uniforms."""
    tok = cost = 0.0
    for i, b in enumerate(betas):
        n, k, _ = replay(u_list[i] < b, gamma_fn(i), c)
        tok += n
        cost += k
    return tok / cost


def mean_thr(betas, U, gamma_fn, c):
    return [throughput(betas, u, gamma_fn, c) for u in U]


def fit_calibrator(top_tr, ent_tr, beta_tr):
    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(top_tr, beta_tr)
    X = np.column_stack([top_tr, ent_tr])
    Xd = np.vstack([X, X])
    yd = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    wd = np.concatenate([beta_tr, 1 - beta_tr])  # soft labels via weights
    lr = LogisticRegression(C=1e4, max_iter=2000).fit(Xd, yd, sample_weight=wd)
    return {
        "isotonic": lambda t, e: iso.predict(t),
        "logistic": lambda t, e: lr.predict_proba(np.column_stack([t, e]))[:, 1],
    }


def acf(x, maxlag=10):
    """Autocorrelation of beta within passages (pooled over passages)."""
    mu, var = x.mean(), x.var()
    return [float(np.mean((x[:, :-k] - mu) * (x[:, k:] - mu)) / var) for k in range(1, maxlag + 1)]


def analyse(pair, typ, beta, top, ent, c_meas, split_seed, replay_seeds, examples):
    n = len(beta)
    perm = np.random.default_rng(split_seed).permutation(n)
    tr, te = np.sort(perm[: n // 2]), np.sort(perm[n // 2 :])
    rows, cal_rows = [], []
    U_all = [np.random.default_rng(s).uniform(size=beta.shape) for s in replay_seeds]
    Utr = [u[tr] for u in U_all]
    Ute = [u[te] for u in U_all]
    btr, bte = beta[tr], beta[te]
    cal = fit_calibrator(top[tr].ravel(), ent[tr].ravel(), btr.ravel())
    m_tr = float(btr.mean())

    for c_label, c in [("measured", c_meas)] + [(str(x), x) for x in C_SENS]:
        res = {}
        res["no_spec"] = (0, mean_thr(bte, Ute, lambda i: 0, c))
        res["fixed_4"] = (4, mean_thr(bte, Ute, lambda i: 4, c))
        kL, sL = best_integer_gamma(m_tr, c)
        res["landmark"] = (kL, mean_thr(bte, Ute, lambda i: kL, c))
        trf = [np.mean(mean_thr(btr, Utr, lambda i, g=g: g, c)) for g in range(GAMMA_MAX + 1)]
        gB = int(np.argmax(trf))
        res["best_fixed"] = (gB, mean_thr(bte, Ute, lambda i: gB, c))

        def ad_thr(beta_sub, hat_sub, U, lam):
            gam = [per_token_gamma(h, lam, c) for h in hat_sub]
            return np.mean(mean_thr(beta_sub, U, lambda i: gam[i], c))

        lam_o, _, conv_o = fit_lambda_fixed_point(lambda l: ad_thr(btr, btr, Utr, l))
        gte = [per_token_gamma(h, lam_o, c) for h in bte]
        res["adaptive_oracle"] = (f"lam={lam_o:.3f}", mean_thr(bte, Ute, lambda i: gte[i], c))

        best_cal = None
        for name, fn in cal.items():
            hat_tr = fn(top[tr].ravel(), ent[tr].ravel()).reshape(btr.shape)
            lam_r, _, conv_r = fit_lambda_fixed_point(lambda l: ad_thr(btr, hat_tr, Utr, l))
            v = ad_thr(btr, hat_tr, Utr, lam_r)
            mse = float(np.mean((hat_tr - btr) ** 2))
            cal_rows.append(dict(pair=pair, type=typ, c_label=c_label, c=c, calibrator=name, train_mse=mse,
                                 train_throughput=v, lam=lam_r, converged=conv_r))
            if best_cal is None or v > best_cal[0]:
                best_cal = (v, name, lam_r)
        _, cname, lam_r = best_cal
        hat_te = cal[cname](top[te].ravel(), ent[te].ravel()).reshape(bte.shape)
        gre = [per_token_gamma(h, lam_r, c) for h in hat_te]
        res["adaptive_realistic"] = (f"{cname}, lam={lam_r:.3f}", mean_thr(bte, Ute, lambda i: gre[i], c))

        for pol, (g, v) in res.items():
            rows.append(dict(pair=pair, type=typ, c_label=c_label, c=c, policy=pol, gamma=str(g),
                             thr_mean=float(np.mean(v)), thr_sd=float(np.std(v, ddof=1))))
        bf = np.mean(res["best_fixed"][1])
        orc = np.mean(res["adaptive_oracle"][1])
        rea = np.mean(res["adaptive_realistic"][1])
        rows[-1]["recovery_pct"] = float(100 * (rea - bf) / (orc - bf)) if orc != bf else float("nan")
        if c_label == "measured":
            # Jensen check on the test half: landmark formula with mean beta vs replay at the same gamma
            m_te = float(bte.mean())
            jensen = dict(pair=pair, type=typ, c=c, gamma=kL, mean_beta_test=m_te,
                          landmark_pred=float(speedup(m_te, kL, c)),
                          replay_mean=float(np.mean(res["landmark"][1])),
                          replay_sd=float(np.std(res["landmark"][1], ddof=1)),
                          lag1_acf_test=acf(bte, 1)[0],
                          gamma_star_real_train=gamma_star(m_tr, c))
            jensen["replay_minus_pred_pct"] = 100 * (jensen["replay_mean"] / jensen["landmark_pred"] - 1)
            # control: same passages, lengths and uniforms, but every beta_t set to the mean beta
            const = mean_thr(np.full_like(bte, m_te), Ute, lambda i: kL, c)
            jensen["const_mean"] = float(np.mean(const))
            jensen["const_sd"] = float(np.std(const, ddof=1))
            jensen["replay_minus_const_pct"] = 100 * (jensen["replay_mean"] / jensen["const_mean"] - 1)
            if typ in ("prose", "code") and pair == "large":
                examples[typ] = dict(beta=bte[0], starts=np.array(replay_boundaries(Ute[0][0] < bte[0], gte[0])),
                                     c=c, lam=lam_o)
    return rows, cal_rows, jensen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="passages per type")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--targets", default="large,medium")
    args = ap.parse_args()
    seeds = dict(split=SEEDS["e3_split"], data=SEEDS["e3_data"], replay=SEEDS["e3_replay"])
    log_seeds("e3", seeds)
    torch.manual_seed(0)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DRAFT)
    passages, counts = load_passages(tok, args.n, seeds["data"])
    (RESULTS / "e3_data_counts.json").write_text(json.dumps(counts, indent=2))
    print("passages:", counts)

    dev = pick_device(args.cpu)
    print("device:", dev)
    draft = load_model(DRAFT, dev)
    prompt = passages["prose"][0][:PREFIX].tolist()

    timing, desc, all_rows, all_cal, jens, examples = [], [], [], [], [], {}
    td = time_model(draft, prompt, dev)
    timing.append(dict(model=DRAFT, role="draft", device=str(dev), per_token_s=td[0], per_token_sd=td[1]))

    for pair in args.targets.split(","):
        name = TARGETS[pair]
        try:
            target = load_model(name, dev)
            bs = 4 if pair == "large" else 8
            traces = {}
            for typ in TYPES:
                traces[typ] = traces_for_pair(draft, target, passages[typ], dev, bs)
            tt = time_model(target, prompt, dev)
            t1, t8 = time_verify(target, prompt, dev)
        except (RuntimeError, MemoryError) as e:
            if pair == "large":
                note_deviation(f"gpt2-large failed on {dev} ({type(e).__name__}: {str(e)[:200]}); using gpt2-medium only")
                del target
                gc.collect()
                continue
            raise
        c_meas = td[0] / tt[0]
        timing.append(dict(model=name, role="target", pair=pair, device=str(dev), per_token_s=tt[0], per_token_sd=tt[1],
                           c=c_meas, verify_1tok_s=t1, verify_8tok_s=t8, verify_ratio_8_over_1=t8 / t1))
        print(f"[{pair}] c = {c_meas:.3f}; verify 8 vs 1 token = {t8 / t1:.2f}x")
        for typ in TYPES:
            b, tp, en = traces[typ]
            np.savez(RESULTS / f"traces_{pair}_{typ}.npz", beta=b, draft_top=tp, draft_entropy=en,
                     token_ids=passages[typ])
            q = np.quantile(b, [0.05, 0.25, 0.5, 0.75, 0.95])
            d = dict(pair=pair, type=typ, n_passages=len(b), mean=b.mean(), sd=b.std(), q05=q[0], q25=q[1],
                     median=q[2], q75=q[3], q95=q[4], frac_gt_0_9=(b > 0.9).mean(), frac_lt_0_3=(b < 0.3).mean())
            for k, v in enumerate(acf(b), 1):
                d[f"acf_lag{k}"] = v
            desc.append(d)
            rows, cal_rows, j = analyse(pair, typ, b, tp, en, c_meas, seeds["split"], seeds["replay"], examples)
            all_rows += rows
            all_cal += cal_rows
            jens.append(j)
            print(f"  {typ}: mean beta {b.mean():.3f}, lag1 acf {d['acf_lag1']:.3f}")
        del target
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    pd.DataFrame(timing).to_csv(RESULTS / "timing.csv", index=False)
    pd.DataFrame(desc).to_csv(RESULTS / "e3_beta_stats.csv", index=False)
    pd.DataFrame(all_rows).to_csv(RESULTS / "e3_replay.csv", index=False)
    pd.DataFrame(all_cal).to_csv(RESULTS / "e3_calibrators.csv", index=False)
    pd.DataFrame(jens).to_csv(RESULTS / "e3_jensen.csv", index=False)
    if examples:
        np.savez(RESULTS / "e3_fig4_example.npz",
                 **{f"{k}_{f}": v[f] for k, v in examples.items() for f in ("beta", "starts", "c", "lam")})
    print(pd.DataFrame(all_rows).query("c_label == 'measured'")[["pair", "type", "policy", "gamma", "thr_mean", "thr_sd"]]
          .to_string(index=False))


if __name__ == "__main__":
    with Timer("e3"):
        main()
