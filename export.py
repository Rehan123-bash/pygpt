"""
export.py - Shrink a training checkpoint (~1.3 GB: fp32 weights + optimizer + scaler)
to an fp16 weights-only file (~220 MB) for the Gradio app / HF Hub.

The app reconstructs the model from model.py + model_args, so no HF conversion is needed.

Usage:
  python export.py --ckpt /kaggle/working/ckpt_best.pt --out app/model_fp16.pt
  python export.py --ckpt runs/smoke/ckpt_best.pt --out app/model_fp16.pt --verify
  # also copy the tokenizer so the app runs offline:
  python export.py --ckpt ... --out app/model_fp16.pt --tokenizer_src /kaggle/input/pygpt-python-tokens/tokenizer
"""

import argparse
import os
import shutil

import torch

from model import GPT, GPTConfig


def export_fp16(ckpt_path, out_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ckpt["model"].items()}
    state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    # lm_head.weight is tied to wte.weight; dropping it saves 50 MB (the app re-ties on load)
    state.pop("lm_head.weight", None)
    slim = {"model": state, "model_args": ckpt["model_args"],
            "iter_num": ckpt["iter_num"], "val_loss": ckpt.get("val_loss"),
            "tokens_seen": ckpt.get("tokens_seen")}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(slim, out_path)
    print(f"{ckpt_path} ({os.path.getsize(ckpt_path)/2**20:.0f} MB) -> "
          f"{out_path} ({os.path.getsize(out_path)/2**20:.0f} MB), step {slim['iter_num']}")
    return slim


def load_exported(path, device="cpu"):
    """Used by the app: rebuild the model from an exported fp16 file (fp32 on CPU for speed)."""
    exp = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**exp["model_args"]))
    state = dict(exp["model"])
    state["lm_head.weight"] = state["transformer.wte.weight"]        # re-tie
    if device == "cpu":
        state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, exp


def verify(ckpt_path, out_path, device="cpu"):
    """Greedy outputs of original and exported model should match on a fixed prompt."""
    from generate import load_model
    torch.manual_seed(0)
    orig, _ = load_model(ckpt_path, device)
    exp, _ = load_exported(out_path, device)
    prompt = torch.randint(0, orig.config.vocab_size, (1, 16))
    a = orig.generate(prompt.clone(), max_new_tokens=20, temperature=0.0)
    b = exp.generate(prompt.clone(), max_new_tokens=20, temperature=0.0)
    same = torch.equal(a, b)
    print(f"verify: greedy 20-token continuation identical = {same}"
          + ("" if same else "  (small fp16 rounding differences; inspect before shipping)"))
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="app/model_fp16.pt")
    ap.add_argument("--tokenizer_src", default=None, help="copy this tokenizer dir to app/tokenizer")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    export_fp16(args.ckpt, args.out)
    if args.tokenizer_src:
        dst = os.path.join(os.path.dirname(args.out) or ".", "tokenizer")
        if os.path.abspath(args.tokenizer_src) != os.path.abspath(dst):
            shutil.copytree(args.tokenizer_src, dst, dirs_exist_ok=True)
            print(f"copied tokenizer -> {dst}")
    if args.verify:
        verify(args.ckpt, args.out)


if __name__ == "__main__":
    main()
