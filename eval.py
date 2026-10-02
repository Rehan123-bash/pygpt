"""
eval.py - Evaluate a trained checkpoint. Implements E1-E5 from PROJECT.md section 4.7:

  E1  val loss + perplexity on held-out tokens (val.bin), vs a random-init baseline
  E2  syntactic validity: % of sampled completions where ast.parse(prompt+completion) succeeds
  E3  mini functional test: 5 tasks x 3 hidden unit tests, pass@1 and pass@10
  E4  loss curve plotted from the training log.csv
  E5  qualitative samples for the 6 demo prompts (Appendix B)

Baselines: random-init model (always, cheap) and vanilla GPT-2 124M (--gpt2, downloads
~500 MB; English-only, so expect low validity - that is the point of the comparison).

Outputs: <out_dir>/metrics.json, loss_curve.png, samples.md

Usage:
  python eval.py --ckpt /kaggle/working/ckpt_best.pt --data_dir /kaggle/input/pygpt-python-tokens --gpt2
  python eval.py --ckpt runs/smoke/ckpt_best.pt --data_dir data/synthetic --fast   # CPU pipeline test
"""

import argparse
import ast as ast_mod
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from model import GPT, GPTConfig
from generate import load_model, complete, trim_completion

# ---------------------------------------------------------------------------
# E2: 20 short prompts covering defs, classes, imports, control flow
# ---------------------------------------------------------------------------
AST_PROMPTS = [
    'def add(a, b):\n',
    'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n',
    'def is_even(n):\n',
    'def count_vowels(s):\n    """Count the vowels in s."""\n',
    'class Point:\n    def __init__(self, x, y):\n',
    'class Counter:\n',
    'import os\n\ndef list_python_files(path):\n',
    'import json\n\ndef load_config(path):\n',
    'def get_max(numbers):\n    """Return the largest number in the list."""\n',
    'def celsius_to_fahrenheit(c):\n',
    'for i in range(10):\n',
    'def merge_dicts(a, b):\n',
    'def flatten(nested):\n    """Flatten a list of lists."""\n',
    'with open("data.txt") as f:\n',
    'def square_all(xs):\n    return [',
    'def safe_divide(a, b):\n    try:\n',
    'import re\n\ndef find_emails(text):\n',
    'def binary_search(arr, target):\n',
    'class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n',
    'def word_frequencies(text):\n    """Return a dict mapping each word to its count."""\n',
]

# ---------------------------------------------------------------------------
# E3: mini functional test. Prompts are written in TheAlgorithms style (type
# hints, docstring, doctest examples) to match the DSA curriculum data.
# ---------------------------------------------------------------------------
FUNC_TASKS = [
    dict(name="is_prime",
         prompt='def is_prime(number: int) -> bool:\n'
                '    """Return True if number is a prime number, else False.\n\n'
                '    >>> is_prime(7)\n    True\n    >>> is_prime(10)\n    False\n    """\n',
         tests=["assert is_prime(2) == True", "assert is_prime(9) == False", "assert is_prime(17) == True"]),
    dict(name="add",
         prompt='def add(a: int, b: int) -> int:\n'
                '    """Return the sum of a and b.\n\n'
                '    >>> add(2, 3)\n    5\n    >>> add(-1, 1)\n    0\n    """\n',
         tests=["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]),
    dict(name="multiply",
         prompt='def multiply(a: int, b: int) -> int:\n'
                '    """Return a multiplied by b.\n\n'
                '    >>> multiply(3, 4)\n    12\n    >>> multiply(-2, 3)\n    -6\n    """\n',
         tests=["assert multiply(3, 4) == 12", "assert multiply(-2, 3) == -6", "assert multiply(0, 5) == 0"]),
    dict(name="divide",
         prompt='def divide(a: float, b: float) -> float:\n'
                '    """Return a divided by b.\n\n'
                '    >>> divide(10, 2)\n    5.0\n    >>> divide(7, 2)\n    3.5\n    """\n',
         tests=["assert divide(10, 2) == 5.0", "assert divide(7, 2) == 3.5", "assert divide(-6, 3) == -2.0"]),
    dict(name="find_median",
         prompt='def find_median(numbers: list) -> float:\n'
                '    """Return the median value of a list of numbers.\n\n'
                '    >>> find_median([3, 1, 2])\n    2\n    >>> find_median([1, 2, 3, 4])\n    2.5\n    """\n',
         tests=["assert find_median([3, 1, 2]) == 2", "assert find_median([1, 2, 3, 4]) == 2.5",
                "assert find_median([7]) == 7"]),
]

# E5: demo prompts = the five functional tasks (A-E) + the honest failure case (F)
DEMO_PROMPTS = [(chr(65 + i), t["prompt"]) for i, t in enumerate(FUNC_TASKS)]
DEMO_PROMPTS.append(("F", 'def solve_sudoku(board):\n'))


# ---------------------------------------------------------------------------
# E1: deterministic val loss over sequential non-overlapping windows
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_loss(model, val_path, batch_size, max_batches, device):
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    T = model.config.block_size
    n_windows = (len(data) - 1) // T
    losses = []
    for b in range(min(max_batches, n_windows // batch_size)):
        starts = [(b * batch_size + j) * T for j in range(batch_size)]
        x = torch.stack([torch.from_numpy(data[s:s + T].astype(np.int64)) for s in starts]).to(device)
        y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + T].astype(np.int64)) for s in starts]).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    mean = float(np.mean(losses))
    return {"loss": round(mean, 4), "perplexity": round(math.exp(mean), 2),
            "tokens_evaluated": len(losses) * batch_size * T}


# ---------------------------------------------------------------------------
# E2: syntactic validity of sampled completions
# ---------------------------------------------------------------------------
def ast_validity(sample_fn, prompts, n_samples):
    ok = total = 0
    for prompt in prompts:
        for _ in range(n_samples):
            code = prompt + trim_completion(sample_fn(prompt))
            total += 1
            try:
                ast_mod.parse(code)
                ok += 1
            except SyntaxError:
                pass
    return {"valid_pct": round(100.0 * ok / max(1, total), 1), "n_samples": total}


# ---------------------------------------------------------------------------
# E3: functional correctness, candidates executed in a subprocess sandbox
# ---------------------------------------------------------------------------
def run_candidate(code, timeout_s):
    """Run candidate code + asserts in a fresh isolated interpreter (-I); pass = exit 0."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cand.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout_s, cwd=d)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def pass_at_k(n, c, k):
    """Unbiased estimator from Chen et al. 2021: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def functional_test(sample_fn, n_samples, timeout_s):
    per_task = {}
    for task in FUNC_TASKS:
        correct = 0
        for _ in range(n_samples):
            body = trim_completion(sample_fn(task["prompt"]))
            code = task["prompt"] + body + "\n\n" + "\n".join(task["tests"]) + "\n"
            if run_candidate(code, timeout_s):
                correct += 1
        per_task[task["name"]] = {"correct": correct, "n": n_samples}
    k = min(10, n_samples)
    return {
        "pass@1": round(float(np.mean([pass_at_k(t["n"], t["correct"], 1) for t in per_task.values()])), 3),
        f"pass@{k}": round(float(np.mean([pass_at_k(t["n"], t["correct"], k) for t in per_task.values()])), 3),
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# E4: loss curve from log.csv
# ---------------------------------------------------------------------------
def plot_loss_curve(log_csv, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as csv_mod

    steps_t, tok_t, tr = [], [], []
    tok_v, va = [], []
    with open(log_csv) as f:
        for row in csv_mod.DictReader(f):
            if row["train_loss"]:
                tok_t.append(int(row["tokens"]) / 1e6); tr.append(float(row["train_loss"]))
            if row["val_loss"]:
                tok_v.append(int(row["tokens"]) / 1e6); va.append(float(row["val_loss"]))
    plt.figure(figsize=(8, 4))
    plt.plot(tok_t, tr, label="train", alpha=0.6)
    if va:
        plt.plot(tok_v, va, "o-", label="val")
    plt.xlabel("tokens (M)"); plt.ylabel("loss (nats/token)")
    plt.legend(); plt.grid(alpha=0.3); plt.title("PyGPT training")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    return {"final_train": tr[-1] if tr else None, "final_val": va[-1] if va else None}


# ---------------------------------------------------------------------------
# E5: qualitative samples
# ---------------------------------------------------------------------------
def write_samples(sample_fn, out_md, n_per_prompt=2):
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# PyGPT samples (Appendix B prompts)\n")
        for label, prompt in DEMO_PROMPTS:
            f.write(f"\n## Prompt {label}\n")
            for i in range(n_per_prompt):
                text = trim_completion(sample_fn(prompt))
                f.write(f"\n```python\n{prompt}{text}\n```\n")


# ---------------------------------------------------------------------------
# Samplers: our model / random init / vanilla GPT-2, all behind prompt -> completion
# ---------------------------------------------------------------------------
def make_sampler(model, tok, args, device):
    def fn(prompt):
        return complete(model, tok, prompt, args.max_new_tokens, args.temperature,
                        args.top_k, args.top_p, device)
    return fn


def make_gpt2_sampler(args, device):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    @torch.no_grad()
    def fn(prompt):
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:])
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True, help="dir with val.bin and tokenizer/")
    ap.add_argument("--tokenizer", default=None, help="override; default <data_dir>/tokenizer or HF name")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--log_csv", default=None, help="training log for the loss curve; default <ckpt dir>/log.csv")
    ap.add_argument("--eval_batches", type=int, default=80, help="E1 batches of 8x512 tokens")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_ast_samples", type=int, default=5, help="E2 samples per prompt")
    ap.add_argument("--n_func_samples", type=int, default=10, help="E3 samples per task")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--exec_timeout", type=float, default=5.0, help="seconds per E3 candidate")
    ap.add_argument("--gpt2", action="store_true", help="also run E2/E3 with vanilla GPT-2 (downloads ~500 MB)")
    ap.add_argument("--fast", action="store_true", help="tiny sample counts: CPU pipeline test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.fast:
        args.eval_batches, args.n_ast_samples, args.n_func_samples, args.max_new_tokens = 4, 1, 2, 32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    model, ckpt = load_model(args.ckpt, device)
    tok_src = args.tokenizer or os.path.join(args.data_dir, "tokenizer")
    if not os.path.isdir(tok_src) and args.tokenizer is None:
        tok_src = "codeparrot/codeparrot"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_src)
    print(f"model: step {ckpt['iter_num']}, {model.num_params()/1e6:.1f}M params | tokenizer {tok_src} | {device}")

    metrics = {"ckpt": {"path": args.ckpt, "iter_num": ckpt["iter_num"],
                        "tokens_seen": ckpt.get("tokens_seen"), "params_M": round(model.num_params() / 1e6, 1)},
               "settings": {k: getattr(args, k) for k in
                            ("n_ast_samples", "n_func_samples", "max_new_tokens", "temperature", "top_k", "top_p")}}

    # E1 - ours and a random-init model of the same architecture
    val_path = os.path.join(args.data_dir, "val.bin")
    print("E1: val loss ...")
    metrics["e1_val"] = {"ours": val_loss(model, val_path, args.batch_size, args.eval_batches, device)}
    rand_model = GPT(GPTConfig(**ckpt["model_args"])).to(device).eval()
    metrics["e1_val"]["random_init"] = val_loss(rand_model, val_path, args.batch_size, min(4, args.eval_batches), device)
    print(f"  ours: loss {metrics['e1_val']['ours']['loss']}, ppl {metrics['e1_val']['ours']['perplexity']} | "
          f"random init: ppl {metrics['e1_val']['random_init']['perplexity']}")

    ours = make_sampler(model, tok, args, device)
    rand = make_sampler(rand_model, tok, args, device)

    # E2 - syntactic validity
    print("E2: ast validity ...")
    metrics["e2_ast_validity"] = {"ours": ast_validity(ours, AST_PROMPTS, args.n_ast_samples),
                                  "random_init": ast_validity(rand, AST_PROMPTS, max(1, args.n_ast_samples // 5))}
    print(f"  ours: {metrics['e2_ast_validity']['ours']['valid_pct']}% | "
          f"random init: {metrics['e2_ast_validity']['random_init']['valid_pct']}%")

    # E3 - functional mini-test
    print("E3: functional test ...")
    metrics["e3_functional"] = {"ours": functional_test(ours, args.n_func_samples, args.exec_timeout)}
    print(f"  ours: {metrics['e3_functional']['ours']}")

    # GPT-2 baseline for E2/E3 (different tokenizer, so no E1 loss comparison)
    if args.gpt2:
        print("baseline: vanilla GPT-2 124M (E2 + E3) ...")
        gpt2 = make_gpt2_sampler(args, device)
        metrics["e2_ast_validity"]["gpt2"] = ast_validity(gpt2, AST_PROMPTS, args.n_ast_samples)
        metrics["e3_functional"]["gpt2"] = functional_test(gpt2, args.n_func_samples, args.exec_timeout)
        print(f"  gpt2: ast {metrics['e2_ast_validity']['gpt2']['valid_pct']}% | {metrics['e3_functional']['gpt2']}")

    # E4 - loss curve
    log_csv = args.log_csv or os.path.join(os.path.dirname(args.ckpt) or ".", "log.csv")
    if os.path.exists(log_csv):
        print(f"E4: plotting {log_csv} ...")
        metrics["e4_curve"] = plot_loss_curve(log_csv, os.path.join(args.out_dir, "loss_curve.png"))
    else:
        print(f"E4: {log_csv} not found, skipping plot")

    # E5 - qualitative samples
    print("E5: writing samples.md ...")
    write_samples(ours, os.path.join(args.out_dir, "samples.md"))

    metrics["eval_minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {args.out_dir}/metrics.json, loss_curve.png, samples.md in {metrics['eval_minutes']} min")


if __name__ == "__main__":
    main()
