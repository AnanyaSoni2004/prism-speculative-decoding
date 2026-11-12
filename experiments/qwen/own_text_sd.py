"""
own_text_sd.py  --  Fix 1 for the PRISM paper (also covers fixes 2, 3, 5 via flags).

What it does
------------
1. OWN-TEXT TRACES. For each prompt, the TARGET model writes its own continuation
   (temperature 1 sampling, or greedy with --temperature 0). One forward pass of the
   target and one of the draft over that text then record, at every generated position t:
       beta_t = sum_x min(p_t(x), q_t(x))       expected acceptance (what the paper used)
       a_t    = min(1, q_t(x_t) / p_t(x_t))     acceptance GIVEN the token x_t that was written
   (greedy: a_t = beta_t = 1 if argmax q_t == x_t else 0).

2. EXACT REPLAY. Replaying Bernoulli(a_t) along target-written text reproduces real
   speculative decoding exactly in distribution, for every gamma, because
       P(token = x, accepted)  = min(p, q)(x)      = p(x) * min(1, q(x)/p(x))
       P(token = x, rejected)  = max(0, p - q)(x)  = p(x) * (1 - min(1, q(x)/p(x))).
   Replaying Bernoulli(beta_t) instead ("beta replay") ignores that accepted tokens are
   the ones the draft likes, so it is only an approximation.

3. END-TO-END CHECK. Real speculative decoding (Leviathan et al., Algorithm 1) is run on a
   subset of prompts and its tokens-per-target-pass is compared with both replays.

4. REFERENCE-TEXT TRACES (--also_reference) reproduce the paper's old method on the same
   passages, so the old and new numbers can be put side by side.

Outputs (in --out): traces.npz, summary.json, and a printed table.

Colab example (T4 GPU):
  pip install -q transformers datasets accelerate
  python own_text_sd.py --target Qwen/Qwen2.5-3B --draft Qwen/Qwen2.5-0.5B \
      --dataset wikitext --n 100 --gen_len 128 --out runs/qwen_prose
  python own_text_sd.py ... --temperature 0          # greedy decoding (fix 5)
  python own_text_sd.py ... --dataset jsonl --prompts_file question.jsonl --chat   # Spec-Bench (fix 3)
"""
import argparse, json, math, os, time
import numpy as np
import torch

# ----------------------------------------------------------------------------- models


def next_token_probs(model, ids, temperature, vocab):
    """Next-token distributions at every position of ids (1, L) -> (L, vocab) float32.
    temperature == 0 returns one-hot argmax distributions (greedy)."""
    logits = model(ids).logits[0, :, :vocab].float()
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), vocab).float()
    return torch.softmax(logits / temperature, dim=-1)


@torch.no_grad()
def sample_target(model, prompt, gen_len, temperature, vocab, gen):
    """Target writes gen_len tokens after prompt (1, P). Uses the KV cache; no top-k/top-p,
    no repetition penalty, no EOS stop, so the text follows softmax(logits / T) exactly."""
    ids, past, inp = prompt, None, prompt
    for _ in range(gen_len):
        out = model(inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :vocab].float()
        if temperature == 0:
            nxt = logits.argmax().view(1, 1)
        else:
            nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=gen).view(1, 1)
        ids = torch.cat([ids, nxt], 1)
        inp = nxt
    return ids


@torch.no_grad()
def trace(target, draft, ids, n_prompt, temperature, vocab):
    """Per-position acceptance quantities for the tokens ids[0, n_prompt:]."""
    p = next_token_probs(target, ids[:, :-1], temperature, vocab)[n_prompt - 1:]
    q = next_token_probs(draft, ids[:, :-1], temperature, vocab)[n_prompt - 1:]
    x = ids[0, n_prompt:]
    px = p.gather(1, x[:, None])[:, 0]
    qx = q.gather(1, x[:, None])[:, 0]
    beta = torch.minimum(p, q).sum(-1)
    a = torch.where(px > 0, torch.clamp(qx / px.clamp_min(1e-30), max=1.0), torch.zeros_like(px))
    # Draft confidence signals for System 1. Under greedy decoding q is one-hot, so they are
    # taken from the draft's temperature-1 distribution instead.
    qf = q if temperature > 0 else next_token_probs(draft, ids[:, :-1], 1.0, vocab)[n_prompt - 1:]
    ent = -(qf * torch.log(qf.clamp_min(1e-30))).sum(-1)
    return dict(beta=beta.cpu().numpy(), a=a.cpu().numpy(),
                qmax=qf.max(-1).values.cpu().numpy(), qent=ent.cpu().numpy())


@torch.no_grad()
def speculative_decode(target, draft, prompt, gen_len, gamma, temperature, vocab, gen):
    """Real speculative decoding (no KV cache: simple and obviously correct; used only as a check).
    Returns (ids, list of tokens produced per target pass)."""
    ids, per_round = prompt, []
    n0 = prompt.shape[1]
    while ids.shape[1] - n0 < gen_len:
        base = ids.shape[1]
        cur, qs = ids, []
        for _ in range(gamma):                                   # draft gamma tokens
            qd = next_token_probs(draft, cur, temperature, vocab)[-1]
            tok = qd.argmax().view(1, 1) if temperature == 0 else torch.multinomial(qd, 1, generator=gen).view(1, 1)
            qs.append(qd)
            cur = torch.cat([cur, tok], 1)
        P = next_token_probs(target, cur, temperature, vocab)[base - 1:]   # one target pass, gamma+1 rows
        n_acc = 0
        for i in range(gamma):
            tok = cur[0, base + i]
            ratio = P[i, tok] / qs[i][tok].clamp_min(1e-30)
            if torch.rand((), generator=gen, device=gen.device).to(ratio.device) < ratio:
                n_acc += 1
            else:
                break
        if n_acc < gamma:                                        # rejected: resample from the residual
            resid = torch.clamp(P[n_acc] - qs[n_acc], min=0)
            resid = resid / resid.sum() if resid.sum() > 0 else P[n_acc]
            nxt = resid.argmax() if temperature == 0 else torch.multinomial(resid, 1, generator=gen)[0]
        else:                                                    # all accepted: bonus token from target
            nxt = P[gamma].argmax() if temperature == 0 else torch.multinomial(P[gamma], 1, generator=gen)[0]
        ids = torch.cat([cur[:, :base + n_acc], nxt.view(1, 1).to(ids.device)], 1)
        per_round.append(n_acc + 1)
    return ids, per_round

# ----------------------------------------------------------------------------- replay


def replay(accept_prob, gamma, rng, n_rep=20):
    """Replay rounds with a fixed gamma along one passage's per-position acceptance probabilities.
    Returns (tokens, rounds) summed over n_rep replays. A round that runs off the end is dropped."""
    T = len(accept_prob)
    tokens = rounds = 0
    for _ in range(n_rep):
        acc = rng.random(T) < accept_prob
        t = 0
        while t + gamma < T:                 # need gamma draft positions + 1 inside the passage
            k = 0
            while k < gamma and acc[t + k]:
                k += 1
            tokens += k + 1
            rounds += 1
            t += k + 1
    return tokens, rounds


def e2e_complete_rounds(per_round, gamma, gen_len):
    """Apply the same end-of-passage rule as replay() to an end-to-end run."""
    tokens = rounds = t = 0
    for n in per_round:
        if t + gamma >= gen_len:
            break
        tokens += n
        rounds += 1
        t += n
    return tokens, rounds


def boot_ci(tok, rnd, rng, B=2000):
    """Ratio-of-sums estimate with a 95% bootstrap CI over passages."""
    tok, rnd = np.asarray(tok, float), np.asarray(rnd, float)
    est = tok.sum() / rnd.sum()
    idx = rng.integers(0, len(tok), (B, len(tok)))
    bs = tok[idx].sum(1) / rnd[idx].sum(1)
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def acf(xs, lag):
    num = den = 0.0
    m = np.mean(np.concatenate(xs))
    for x in xs:
        num += ((x[:-lag] - m) * (x[lag:] - m)).sum()
        den += ((x - m) ** 2).sum()
    return num / den if den > 0 else float("nan")


def summarise(traces, gammas, c, seed, e2e=None, gen_len=None):
    """traces: list of dicts with 'beta' and 'a'. Returns a summary dict."""
    rng = np.random.default_rng(seed)
    betas = [t["beta"] for t in traces]
    allb = np.concatenate(betas)
    alpha = float(allb.mean())
    out = dict(n_passages=len(traces), alpha=alpha, sd_beta=float(allb.std()),
               acf1=float(acf(betas, 1)), acf5=float(acf(betas, 5)), rows=[])
    for g in gammas:
        row = dict(gamma=g, landmark=(1 - alpha ** (g + 1)) / (1 - alpha))
        for name, key in (("exact", "a"), ("beta", "beta")):
            tr = [replay(t[key], g, rng) for t in traces]
            row[name] = boot_ci([x[0] for x in tr], [x[1] for x in tr], rng)
        if e2e and g in e2e:
            cr = [e2e_complete_rounds(pr, g, gen_len) for pr in e2e[g]]
            row["e2e"] = boot_ci([x[0] for x in cr], [x[1] for x in cr], rng)
        row["speedup_exact_A2"] = row["exact"][0] / (g * c + 1) if c else None
        out["rows"].append(row)
    return out


def print_summary(title, s):
    print(f"\n== {title}: {s['n_passages']} passages, mean beta {s['alpha']:.3f}, SD {s['sd_beta']:.3f}, "
          f"ACF lag1/lag5 {s['acf1']:.2f}/{s['acf5']:.2f}")
    print("gamma  landmark  exact replay [95% CI]      beta replay [95% CI]       end-to-end [95% CI]")
    f = lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
    for r in s["rows"]:
        print(f"{r['gamma']:>5}  {r['landmark']:8.3f}  {f(r['exact']):<25}  {f(r['beta']):<25}  "
              f"{f(r['e2e']) if 'e2e' in r else '-'}")

# ----------------------------------------------------------------------------- data


def load_prompts(args, tok):
    """Returns a list of (prompt_ids, reference_ids or None) as 1-D python lists."""
    texts = []
    if args.dataset == "jsonl":                     # Spec-Bench question.jsonl or any {"turns":[..]} / {"prompt":..}
        mt = {"writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"}
        seen = {}
        for line in open(args.prompts_file):
            d = json.loads(line)
            cat = d.get("category", "all")
            cat = "mt_bench" if cat in mt else cat          # Spec-Bench's multi-turn set (first turn only)
            if args.per_category and seen.get(cat, 0) >= args.per_category:
                continue
            seen[cat] = seen.get(cat, 0) + 1
            texts.append((d["turns"][0] if "turns" in d else d.get("prompt") or d["question"], None, cat))
    else:
        from datasets import load_dataset
        if args.dataset == "wikitext":
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
            for t in ds["text"]:
                if len(t.split()) > 200 and not t.strip().startswith("="):
                    texts.append((t.strip(), "split"))
        elif args.dataset == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="test")
            texts = [(f"Question: {q}\nAnswer:", " " + a) for q, a in zip(ds["question"], ds["answer"])]
        elif args.dataset == "humaneval":
            ds = load_dataset("openai/openai_humaneval", split="test")
            texts = [(p, s) for p, s in zip(ds["prompt"], ds["canonical_solution"])]
    out = []
    for item in texts:
        prompt, ref, cat = item if len(item) == 3 else (*item, args.dataset)
        if args.chat:
            p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
            p_ids = list(p_ids["input_ids"]) if hasattr(p_ids, "keys") else list(p_ids)
            r_ids = None
            if len(p_ids) > args.max_prompt:                 # too long for the GPU: skip, never truncate
                continue
        elif ref == "split":                         # running prose: first prompt_len tokens are the prompt
            ids = tok(prompt)["input_ids"]
            if len(ids) < args.prompt_len + args.gen_len:
                continue
            p_ids, r_ids = ids[:args.prompt_len], ids[args.prompt_len:args.prompt_len + args.gen_len]
        else:
            p_ids = tok(prompt)["input_ids"][-args.max_prompt:]
            r_ids = tok(ref, add_special_tokens=False)["input_ids"][:args.gen_len] if ref else None
        out.append((p_ids, r_ids, cat))
        if len(out) == args.n and not args.per_category:
            break
    return out

# ----------------------------------------------------------------------------- main


def run(target, draft, prompts, args, device, vocab):
    gen = torch.Generator(device=device).manual_seed(args.seed)
    own, ref, e2e = [], [], {g: [] for g in args.e2e_gammas}
    t0 = time.time()
    for i, (p_ids, r_ids, _cat) in enumerate(prompts):
        prompt = torch.tensor([p_ids], device=device)
        ids = sample_target(target, prompt, args.gen_len, args.temperature, vocab, gen)
        own.append(trace(target, draft, ids, len(p_ids), args.temperature, vocab))
        if args.also_reference and r_ids and len(r_ids) >= 16:
            rid = torch.tensor([p_ids + r_ids], device=device)
            ref.append(trace(target, draft, rid, len(p_ids), args.temperature, vocab))
        if i < args.e2e_n:
            for g in args.e2e_gammas:
                _, pr = speculative_decode(target, draft, prompt, args.gen_len, g, args.temperature, vocab, gen)
                e2e[g].append(pr)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts, {time.time() - t0:.0f}s", flush=True)
    return own, ref, e2e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--dataset", default="wikitext", choices=["wikitext", "gsm8k", "humaneval", "jsonl"])
    ap.add_argument("--prompts_file")
    ap.add_argument("--chat", action="store_true", help="wrap prompts in the tokenizer's chat template")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per_category", type=int, default=0, help="jsonl only: prompts per category (0 = first --n lines)")
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--max_prompt", type=int, default=512)
    ap.add_argument("--gen_len", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0, help="0 = greedy")
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    ap.add_argument("--e2e_gammas", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--e2e_n", type=int, default=30, help="prompts used for the end-to-end check")
    ap.add_argument("--c", type=float, default=None, help="draft/target time ratio, for speedups under (A2)")
    ap.add_argument("--also_reference", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/own_text")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"auto": torch.float16 if device == "cuda" else torch.float32, "float32": torch.float32,
             "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=dtype).to(device).eval()
    # Same-family models can pad the embedding table differently (e.g. Qwen2.5 0.5B vs 7B).
    vocab = min(target.config.vocab_size, draft.config.vocab_size, len(tok))
    print(f"device {device}, dtype {dtype}, shared vocab {vocab}")

    prompts = load_prompts(args, tok)
    print(f"{len(prompts)} prompts loaded")
    own, ref, e2e = run(target, draft, prompts, args, device, vocab)

    os.makedirs(args.out, exist_ok=True)
    s_own = summarise(own, args.gammas, args.c, args.seed, e2e, args.gen_len)
    print_summary("OWN TEXT (target-written)", s_own)
    result = dict(args=vars(args), own=s_own)
    cats = [c for _, _, c in prompts]
    if len(set(cats)) > 1:                                   # one table per category (Spec-Bench)
        result["by_category"] = {}
        for c in sorted(set(cats)):
            ix = [i for i, x in enumerate(cats) if x == c]
            sc = summarise([own[i] for i in ix], args.gammas, args.c, args.seed,
                           {g: [e2e[g][i] for i in ix if i < len(e2e[g])] for g in e2e} if args.e2e_n >= len(prompts) else None,
                           args.gen_len)
            print_summary(f"category {c}", sc)
            result["by_category"][c] = sc
    if ref:
        s_ref = summarise(ref, args.gammas, args.c, args.seed)
        print_summary("REFERENCE TEXT (old method)", s_ref)
        result["reference"] = s_ref
    json.dump(result, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(args.out, "traces.npz"), categories=np.array(cats),
                        **{f"own_{k}": np.stack([t[k] for t in own]) for k in own[0]},
                        **({f"ref_{k}": np.array([t[k] for t in ref], dtype=object) for k in ref[0]} if ref else {}))
    print(f"\nsaved to {args.out}/")


if __name__ == "__main__":
    main()
