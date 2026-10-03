"""
app.py - Gradio code-completion demo for PyGPT (PROJECT.md section 4.8).

Needs, in the same folder as this file (or via env vars):
    model_fp16.pt   exported weights  (python export.py --ckpt ckpt_best.pt --out app/model_fp16.pt)
    tokenizer/      saved codeparrot tokenizer (prepare_data.py saves one; export.py --tokenizer_src copies it)

Runs in three places with the same file:
    python app/app.py                    # laptop fallback, CPU is fine (~10 tok/s)
    Hugging Face Space                   # upload app/* plus model.py, generate.py, export.py as the repo root
    Kaggle                               # GRADIO_SHARE=1 python app/app.py -> temporary public link

Env overrides: MODEL_PATH, TOKENIZER_PATH, GRADIO_SHARE=1.
"""

import ast
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]          # find model/generate/export both locally and on Spaces

from export import load_exported
from generate import complete, trim_completion

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "model_fp16.pt"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", os.path.join(HERE, "tokenizer"))

# TheAlgorithms-style prompts (type hints + docstring + doctest) — matches the
# DSA curriculum phase of training, so these are the model's home turf.
EXAMPLES = [
    'def is_prime(number: int) -> bool:\n    """Return True if number is a prime number, else False.\n\n'
    '    >>> is_prime(7)\n    True\n    >>> is_prime(10)\n    False\n    """',
    'def add(a: int, b: int) -> int:\n    """Return the sum of a and b.\n\n'
    '    >>> add(2, 3)\n    5\n    """',
    'def multiply(a: int, b: int) -> int:\n    """Return a multiplied by b.\n\n'
    '    >>> multiply(3, 4)\n    12\n    """',
    'def divide(a: float, b: float) -> float:\n    """Return a divided by b.\n\n'
    '    >>> divide(10, 2)\n    5.0\n    """',
    'def find_median(numbers: list) -> float:\n    """Return the median value of a list of numbers.\n\n'
    '    >>> find_median([3, 1, 2])\n    2\n    >>> find_median([1, 2, 3, 4])\n    2.5\n    """',
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, info = load_exported(MODEL_PATH, device)
if not os.path.isdir(TOKENIZER_PATH):
    TOKENIZER_PATH = "codeparrot/codeparrot"           # falls back to downloading from HF
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

model_card = (f"GPT-2 Small architecture, {model.num_params()/1e6:.0f}M params, trained from scratch on Python "
              f"({(info.get('tokens_seen') or 0)/1e6:.0f}M tokens, step {info['iter_num']}"
              + (f", val loss {info['val_loss']:.3f}" if info.get("val_loss") else "") + f") · running on {device}")

# warm-up so the first button click is not slow
complete(model, tok, "def f(x):\n", max_new_tokens=4, temperature=0.0, device=device)


def complete_code(prompt, max_new_tokens, temperature, top_k, top_p, syntax_filter=True):
    """Syntax-filtered sampling: draw up to 3 completions, return the first whose
    prompt+completion parses with ast.parse. Parsing is NOT correctness - it only
    filters out token salad; the functional pass rate is reported separately."""
    if not prompt.strip():
        return "", "write or pick a prompt first"
    t0 = time.time()
    tries = 3 if syntax_filter else 1
    text, note = None, ""
    for i in range(tries):
        cand = trim_completion(complete(model, tok, prompt, int(max_new_tokens),
                                        float(temperature), int(top_k), float(top_p), device))
        text = cand
        try:
            ast.parse(prompt + cand)
            note = f" · sample {i+1}/{tries} parsed" if syntax_filter else ""
            break
        except SyntaxError:
            note = f" · none of {tries} samples parsed" if syntax_filter else ""
    n_tok = len(tok(text)["input_ids"])
    dt = time.time() - t0
    return prompt + text, f"{n_tok} tokens in {dt:.1f} s ({n_tok/max(dt, 1e-9):.1f} tok/s){note}"


import gradio as gr

with gr.Blocks(title="PyGPT — Python code completion") as demo:
    gr.Markdown("# PyGPT — Python code completion\n"
                "A language model trained **from scratch** for a course project. " + model_card)
    with gr.Row():
        with gr.Column():
            prompt_box = gr.Code(label="Prompt (Python)", language="python", lines=10,
                                 value=EXAMPLES[0])
            btn = gr.Button("Complete", variant="primary")
            with gr.Accordion("Sampling settings", open=False):
                max_new = gr.Slider(32, 256, value=96, step=16, label="max new tokens")
                temp = gr.Slider(0.1, 1.2, value=0.4, step=0.05, label="temperature")
                top_k = gr.Slider(0, 200, value=50, step=10, label="top-k (0 = off)")
                top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="top-p")
                syn_filter = gr.Checkbox(value=True, label="syntax-filtered sampling (best of 3)")
        with gr.Column():
            out_box = gr.Code(label="Prompt + completion", language="python", lines=16)
            stats = gr.Markdown("")
    gr.Examples(examples=EXAMPLES, inputs=prompt_box, label="Example prompts")
    btn.click(complete_code, [prompt_box, max_new, temp, top_k, top_p, syn_filter], [out_box, stats])

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE") == "1")
