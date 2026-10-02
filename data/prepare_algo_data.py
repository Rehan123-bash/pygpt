"""
prepare_algo_data.py - Build the "DSA curriculum" token set for the final phase of
training: textbook algorithm implementations in TheAlgorithms style (type hints,
docstrings, doctests). Two sources, same binary format as prepare_data.py:

  1. TheAlgorithms/Python (MIT) - cloned and tokenized whole
  2. codeparrot-clean files that define classic DSA functions (is_prime, gcd,
     factorial, median, binary_search, ...) - streamed and filtered by name

    python data/prepare_algo_data.py --out_dir data/algo --tokenizer app/tokenizer \
        --stream_tokens 15_000_000

Why: the 800M-token base corpus is random GitHub Python; classic one-function
algorithms are rare in it. Continuing training on a curated ~15-20M tokens for
~1 GPU-hour sharpens exactly the prompts the demo and functional test use.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import keep_file

ALGO_REPO = "https://github.com/TheAlgorithms/Python"          # MIT license

# a codeparrot file is "DSA-like" if it defines any of these
DSA_DEF_RE = re.compile(
    r"def (is_prime|gcd|lcm|factorial|fibonacci|median|find_median|mean|binary_search|"
    r"linear_search|bubble_sort|merge_sort|quick_sort|insertion_sort|selection_sort|"
    r"reverse_string|is_palindrome|two_sum|max_subarray|sum_of_digits|collatz|"
    r"prime_factors|sieve|armstrong|perfect_square|power|add|subtract|multiply|divide)\s*\(")


def tokenize_repo(tok, eos, train_buf, val_buf, val_every):
    n = 0
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "algos")
        print(f"cloning {ALGO_REPO} ...")
        subprocess.run(["git", "clone", "--depth", "1", "--quiet", ALGO_REPO, dst], check=True)
        for root, dirs, files in os.walk(dst):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                try:
                    text = open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if not keep_file(text, max_chars=100_000):
                    continue
                ids = tok(text, add_special_tokens=False)["input_ids"] + [eos]
                (val_buf if n % val_every == 0 else train_buf).extend(ids)
                n += 1
    return n


def stream_dsa_files(tok, eos, train_buf, val_buf, val_every, target_tokens, log_every=25):
    """Stream codeparrot-clean and keep only files that define classic DSA functions."""
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)
    n = kept_tokens = seen = 0
    t0 = time.time()
    for ex in ds:
        seen += 1
        text = ex.get("content") or ""
        if not keep_file(text, max_chars=100_000) or not DSA_DEF_RE.search(text):
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"] + [eos]
        (val_buf if n % val_every == 0 else train_buf).extend(ids)
        n += 1
        kept_tokens += len(ids)
        if n % log_every == 0:
            rate = kept_tokens / max(1e-9, time.time() - t0)
            print(f"  {kept_tokens/1e6:6.1f}M DSA tokens | {n:,} files of {seen:,} seen "
                  f"| ETA {(target_tokens-kept_tokens)/max(rate,1)/60:.0f} min", flush=True)
        if kept_tokens >= target_tokens:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/algo")
    ap.add_argument("--tokenizer", default="app/tokenizer")
    ap.add_argument("--stream_tokens", type=int, default=15_000_000,
                    help="codeparrot DSA tokens to collect on top of TheAlgorithms")
    ap.add_argument("--val_every", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id

    train_buf, val_buf = [], []
    n_repo = tokenize_repo(tok, eos, train_buf, val_buf, args.val_every)
    print(f"TheAlgorithms: {n_repo} files, {len(train_buf)/1e6:.1f}M train tokens so far")
    n_stream = stream_dsa_files(tok, eos, train_buf, val_buf, args.val_every, args.stream_tokens)

    np.asarray(train_buf, dtype=np.uint16).tofile(os.path.join(args.out_dir, "train.bin"))
    np.asarray(val_buf, dtype=np.uint16).tofile(os.path.join(args.out_dir, "val.bin"))
    json.dump({"vocab_size": tok.vocab_size, "eos_id": eos,
               "sources": [ALGO_REPO, "codeparrot-clean-train filtered by DSA_DEF_RE"],
               "train_tokens": len(train_buf), "val_tokens": len(val_buf),
               "repo_files": n_repo, "stream_files": n_stream},
              open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"total: {len(train_buf)/1e6:.1f}M train + {len(val_buf)/1e6:.1f}M val tokens "
          f"({n_repo} repo + {n_stream} streamed files)")


if __name__ == "__main__":
    main()
