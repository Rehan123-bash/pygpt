"""
make_synthetic_data.py - Write small train.bin / val.bin files with a learnable
structure, so train.py and the smoke config can be exercised without downloading
real data. Each token is drawn from a sparse random Markov chain over 256 ids,
so a model can drive the loss from ln(32768) ~ 10.4 down to ~1.1 (= ln 3).

Usage: python tests/make_synthetic_data.py --out_dir data/synthetic --train_tokens 500000
"""

import argparse
import json
import os

import numpy as np


def markov_stream(n_tokens, n_states=256, branching=3, walk_seed=0, chain_seed=12345):
    # the chain (transition table + token ids) is shared by train and val; only the walk differs
    chain = np.random.default_rng(chain_seed)
    table = chain.integers(0, n_states, size=(n_states, branching))   # next-state choices
    ids = chain.permutation(32768)[:n_states]                          # map states to random token ids
    rng = np.random.default_rng(walk_seed)
    choices = rng.integers(0, branching, size=n_tokens)
    out = np.empty(n_tokens, dtype=np.uint16)
    s = 0
    for i in range(n_tokens):
        out[i] = ids[s]
        s = table[s, choices[i]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/synthetic")
    ap.add_argument("--train_tokens", type=int, default=500_000)
    ap.add_argument("--val_tokens", type=int, default=50_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    markov_stream(args.train_tokens, walk_seed=0).tofile(os.path.join(args.out_dir, "train.bin"))
    markov_stream(args.val_tokens, walk_seed=1).tofile(os.path.join(args.out_dir, "val.bin"))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"vocab_size": 32768, "synthetic": True,
                   "train_tokens": args.train_tokens, "val_tokens": args.val_tokens}, f)
    print(f"wrote synthetic data to {args.out_dir}")


if __name__ == "__main__":
    main()
