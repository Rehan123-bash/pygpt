"""
prepare_algo_data.py - Build a small "algorithms curriculum" token set for the final
phase of training: textbook DSA implementations (is_prime, median, sorting, ...)
from MIT-licensed repos. Same binary format as prepare_data.py.

    python data/prepare_algo_data.py --out_dir data/algo --tokenizer app/tokenizer

Why: the 800M-token base corpus is random GitHub Python; classic one-function
algorithms are rare in it. Continuing training on ~5-10M curated tokens for a few
hundred steps sharpens exactly the prompts the demo uses, at negligible cost.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import keep_file

REPOS = [  # (url, license)
    ("https://github.com/TheAlgorithms/Python", "MIT"),
    ("https://github.com/keon/algorithms", "MIT"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/algo")
    ap.add_argument("--tokenizer", default="app/tokenizer")
    ap.add_argument("--val_every", type=int, default=20, help="every Nth file goes to val.bin")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id

    train_buf, val_buf = [], []
    n_files = 0
    with tempfile.TemporaryDirectory() as td:
        for url, lic in REPOS:
            dst = os.path.join(td, url.rsplit("/", 1)[1])
            print(f"cloning {url} ({lic}) ...")
            subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, dst], check=True)
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
                    (val_buf if n_files % args.val_every == 0 else train_buf).extend(ids)
                    n_files += 1

    np.asarray(train_buf, dtype=np.uint16).tofile(os.path.join(args.out_dir, "train.bin"))
    np.asarray(val_buf, dtype=np.uint16).tofile(os.path.join(args.out_dir, "val.bin"))
    import json
    json.dump({"vocab_size": tok.vocab_size, "eos_id": eos, "sources": [u for u, _ in REPOS],
               "train_tokens": len(train_buf), "val_tokens": len(val_buf), "files": n_files},
              open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"{n_files} files -> {len(train_buf)/1e6:.1f}M train + {len(val_buf)/1e6:.1f}M val tokens")
    print(tok.decode(train_buf[:150]))


if __name__ == "__main__":
    main()
