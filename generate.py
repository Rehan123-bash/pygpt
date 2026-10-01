"""
generate.py - Sample completions from a trained checkpoint.

Usage:
    python generate.py --ckpt runs/main/ckpt_best.pt --prompt "def fibonacci(n):"
    python generate.py --ckpt ckpt_best.pt --prompt_file prompt.py --temperature 0.4 --top_k 50
    python generate.py --ckpt ckpt_best.pt --interactive

Loads the tokenizer from --tokenizer (a HF name or a local folder saved by prepare_data.py).
"""

import argparse
import sys

import torch

from model import GPT, GPTConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt


def trim_completion(text: str) -> str:
    """
    Stop a completion at a natural boundary: when a new top-level def/class/decorator
    starts after the body began, or drop a trailing partial line.
    """
    lines = text.split("\n")
    out = []
    body_started = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i > 0 and stripped and not line.startswith((" ", "\t")) and (
            stripped.startswith(("def ", "class ", "@", "if __name__")) and body_started
        ):
            break
        if stripped:
            body_started = body_started or line.startswith((" ", "\t"))
        out.append(line)
    # drop a trailing partial line (generation stopped mid-line)
    if len(out) > 1 and out[-1].strip() and not text.endswith("\n"):
        out = out[:-1]
    return "\n".join(out)


def complete(model, tok, prompt, max_new_tokens=96, temperature=0.6, top_k=50, top_p=0.95, device="cpu"):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    ids = ids[:, -model.config.block_size:]
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, stop_token_id=tok.eos_token_id)
    new_ids = out[0, ids.shape[1]:].tolist()
    if tok.eos_token_id in new_ids:
        new_ids = new_ids[:new_ids.index(tok.eos_token_id)]
    return tok.decode(new_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="codeparrot/codeparrot")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=1, help="samples per prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_trim", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model, ckpt = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt}: step {ckpt['iter_num']}, val loss {ckpt.get('val_loss')}, "
          f"{model.num_params()/1e6:.1f}M params, device {device}", file=sys.stderr)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def run(prompt):
        for i in range(args.n):
            text = complete(model, tok, prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p, device)
            if not args.no_trim:
                text = trim_completion(text)
            print(f"----- sample {i+1} -----")
            print(prompt + text)

    if args.interactive:
        print("Enter a prompt; finish with an empty line. Ctrl-D to quit.", file=sys.stderr)
        while True:
            lines = []
            try:
                while True:
                    line = input()
                    if line == "" and lines:
                        break
                    lines.append(line)
            except EOFError:
                break
            run("\n".join(lines) + "\n")
    else:
        if args.prompt_file:
            prompt = open(args.prompt_file).read()
        elif args.prompt is not None:
            prompt = args.prompt.replace("\\n", "\n")
        else:
            prompt = "def fibonacci(n):\n"
        run(prompt)


if __name__ == "__main__":
    main()
