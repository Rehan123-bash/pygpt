"""
sanity.py - Unit sanity checks for model.py. Run BEFORE any long training run.

    python tests/sanity.py            # ~30 s on CPU

Checks (each maps to a "Verified by" cell in PROJECT.md section 6.2):
  1. init loss ~ ln(vocab_size)            -> the head starts near-uniform, as designed
  2. causal mask                           -> changing token t+1 never changes logits at <= t
  3. weight tying                          -> wte.weight is the same tensor as lm_head.weight
  4. overfit one batch                     -> loss < 0.1 in <= 300 steps (loop + grads + optimizer all work)
  5. generate                              -> shapes, stop token, greedy determinism
  6. parameter count of the main config    -> ~111M for GPT-2 Small arch with 32k vocab
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import GPT, GPTConfig

torch.manual_seed(0)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


small = GPTConfig(vocab_size=32768, block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False)

# 1. init loss
m = GPT(small)
x = torch.randint(0, small.vocab_size, (4, 64))
y = torch.randint(0, small.vocab_size, (4, 64))
_, loss = m(x, y)
expected = math.log(small.vocab_size)
check("init loss ~ ln(V)", abs(loss.item() - expected) < 0.5, f"loss {loss.item():.3f}, ln(V) {expected:.3f}")

# 2. causal mask
m.eval()
with torch.no_grad():
    logits_a, _ = m(x, y)                       # full logits (targets given)
    x2 = x.clone()
    x2[:, 40:] = torch.randint(0, small.vocab_size, (4, 24))   # change the future
    logits_b, _ = m(x2, y)
same_past = torch.allclose(logits_a[:, :40], logits_b[:, :40], atol=1e-5)
diff_future = not torch.allclose(logits_a[:, 40:], logits_b[:, 40:], atol=1e-5)
check("causal mask", same_past and diff_future, f"past unchanged={same_past}, future changed={diff_future}")

# 3. weight tying
check("weight tying", m.transformer.wte.weight.data_ptr() == m.lm_head.weight.data_ptr())

# 4. overfit one batch
m = GPT(small)
m.train()
opt = m.configure_optimizers(0.0, 1e-3, (0.9, 0.95), "cpu")
xb = torch.randint(0, small.vocab_size, (4, 64))
yb = torch.randint(0, small.vocab_size, (4, 64))
t0 = time.time()
final = None
for step in range(300):
    _, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    final = loss.item()
    if final < 0.1:
        break
check("overfit one batch", final < 0.1, f"loss {final:.4f} after {step+1} steps, {time.time()-t0:.1f}s")

# 5. generate
m.eval()
prompt = torch.randint(0, small.vocab_size, (2, 10))
out = m.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50, top_p=0.95)
shape_ok = out.shape == (2, 30) and torch.equal(out[:, :10], prompt)
g1 = m.generate(prompt, max_new_tokens=15, temperature=0.0)
g2 = m.generate(prompt, max_new_tokens=15, temperature=0.0)
greedy_ok = torch.equal(g1, g2)
# stop token: force the greedy next token to be the stop id and check it stops after 1 token
stop_id = int(g1[0, 10])
g3 = m.generate(prompt[:1], max_new_tokens=15, temperature=0.0, stop_token_id=stop_id)
stop_ok = g3.shape[1] == 11
# prompt longer than block_size must be cropped, not crash
long_prompt = torch.randint(0, small.vocab_size, (1, 100))
g4 = m.generate(long_prompt, max_new_tokens=3, temperature=0.0)
crop_ok = g4.shape[1] == 103
check("generate", shape_ok and greedy_ok and stop_ok and crop_ok,
      f"shape={shape_ok} greedy_deterministic={greedy_ok} stop_token={stop_ok} long_prompt_crop={crop_ok}")

# 6. main-config parameter count
main_cfg = GPTConfig(vocab_size=32768, block_size=512, n_layer=12, n_head=12, n_embd=768)
big = GPT(main_cfg)
n = big.num_params() / 1e6
check("main config param count ~111M", 105 < n < 118, f"{n:.1f}M")
del big

print()
print(f"{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
