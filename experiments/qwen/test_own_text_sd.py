"""Checks own_text_sd.py on tiny randomly initialised GPT-2 models (no downloads).
1. Speculative decoding output follows the target distribution (first two tokens, exact enumeration).
2. Exact replay matches end-to-end tokens per round; beta replay is compared alongside."""
import sys, types
import numpy as np, torch
from transformers import GPT2Config, GPT2LMHeadModel
import own_text_sd as S

torch.manual_seed(0)
V = 12
mk = lambda seed, scale: (torch.manual_seed(seed), GPT2LMHeadModel(GPT2Config(
    vocab_size=V, n_positions=96, n_embd=32, n_layer=2, n_head=2, initializer_range=scale)).eval())[1]
target, draft = mk(1, 0.6), mk(2, 0.6)
gen = torch.Generator().manual_seed(0)
prompt = torch.tensor([[1, 2, 3]])

# --- 1. exactness: law of the first two generated tokens under SD vs the target's own law
with torch.no_grad():
    p1 = S.next_token_probs(target, prompt, 1.0, V)[-1]
    joint = torch.zeros(V, V)
    for a in range(V):
        joint[a] = p1[a] * S.next_token_probs(target, torch.cat([prompt, torch.tensor([[a]])], 1), 1.0, V)[-1]
N = 40000
cnt = torch.zeros(V, V)
for _ in range(N):
    ids, _ = S.speculative_decode(target, draft, prompt, 2, 3, 1.0, V, gen)
    cnt[ids[0, 3], ids[0, 4]] += 1
tv = 0.5 * (cnt / N - joint).abs().sum().item()
noise = 0.5 * (torch.distributions.Multinomial(N, joint.flatten()).sample() / N - joint.flatten()).abs().sum().item()
print(f"1. TV(SD output, target) over 2 tokens = {tv:.4f}   (pure sampling noise at this N: {noise:.4f})")

# --- 2. exact replay vs end-to-end
args = types.SimpleNamespace(gen_len=48, temperature=1.0, seed=0, also_reference=False,
                             e2e_gammas=[2, 4], e2e_n=10**9)
prompts = [([int(t) for t in torch.randint(0, V, (4,))], None, "test") for _ in range(1500)]
own, _, e2e = S.run(target, draft, prompts, args, "cpu", V)
S.print_summary("tiny random models", S.summarise(own, [2, 4], None, 0, e2e, args.gen_len))
