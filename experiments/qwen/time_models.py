"""
time_models.py -- measure the real costs that the speedup formula needs.

  c     = time of one draft step / time of one target step        (both with KV cache, 1 new token)
  v(m)  = time for the target to verify m new tokens in one pass / time of one target step

The paper's assumption (A2) is v(m) = 1 for every m. The wall-clock time of a round with
gamma draft tokens is then  T = gamma * c + v(gamma + 1)  target steps.

Colab example:
  python time_models.py --target Qwen/Qwen2.5-3B --draft Qwen/Qwen2.5-0.5B --out runs/timing.json
"""
import argparse, copy, json, time
import numpy as np
import torch


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def time_forward(model, ctx, new, device, reps, warmup=5):
    """Median/IQR time (ms) of a forward pass over `new` (1, m) given a KV cache for `ctx` (1, L).
    The cache is rebuilt as a fresh copy before every call, outside the timed region."""
    base = model(ctx, use_cache=True).past_key_values
    ts = []
    for i in range(reps + warmup):
        past = copy.deepcopy(base)
        sync(device)
        t0 = time.perf_counter()
        model(new, past_key_values=past, use_cache=True)
        sync(device)
        if i >= warmup:
            ts.append(1000 * (time.perf_counter() - t0))
        del past
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 25)), float(np.percentile(ts, 75))


def measure(target, draft, vocab, device, ctx_len, max_m, reps, seed=0):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randint(1000 if vocab > 2000 else 0, vocab, (1, ctx_len), generator=g).to(device)
    new = torch.randint(1000 if vocab > 2000 else 0, vocab, (1, max_m), generator=g).to(device)
    d = time_forward(draft, ctx, new[:, :1], device, reps)
    ver = {m: time_forward(target, ctx, new[:, :m], device, reps) for m in range(1, max_m + 1)}
    t = ver[1]
    return dict(ctx_len=ctx_len, draft_ms=d, target_ms=t, c=d[0] / t[0],
                verify_ms={m: v for m, v in ver.items()}, v={m: v[0] / t[0] for m, v in ver.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--ctx_lens", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--max_m", type=int, default=13, help="largest verify block (gamma + 1)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--out", default="timing.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"auto": torch.float16 if device == "cuda" else torch.float32, "float32": torch.float32,
             "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=dtype).to(device).eval()
    vocab = min(target.config.vocab_size, draft.config.vocab_size)
    name = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"device {name}, dtype {dtype}")

    res = dict(args=vars(args), device=name, runs=[])
    for L in args.ctx_lens:
        r = measure(target, draft, vocab, device, L, args.max_m, args.reps)
        res["runs"].append(r)
        print(f"\ncontext {L} tokens: draft step {r['draft_ms'][0]:.1f} ms [IQR {r['draft_ms'][1]:.1f}-{r['draft_ms'][2]:.1f}], "
              f"target step {r['target_ms'][0]:.1f} ms [IQR {r['target_ms'][1]:.1f}-{r['target_ms'][2]:.1f}],  c = {r['c']:.3f}")
        print("  m (tokens verified):  " + "  ".join(f"{m:>5}" for m in r["v"]))
        print("  v(m) = verify/step:   " + "  ".join(f"{v:5.2f}" for v in r["v"].values()))
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
