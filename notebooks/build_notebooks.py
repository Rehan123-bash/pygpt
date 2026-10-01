"""
build_notebooks.py - Generate the Kaggle notebooks from Python source so they stay
in sync with the repo. Run: python notebooks/build_notebooks.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


GET_CODE = '''
# --- Get the code into /kaggle/working/pygpt ---------------------------------
# Option A: public GitHub repo (set REPO_URL). Option B: a Kaggle Dataset named
# "pygpt-code" containing the repo folder (upload the zip -> Datasets -> New Dataset).
REPO_URL = ""            # e.g. "https://github.com/<you>/pygpt.git"
import os, shutil, subprocess
os.chdir("/kaggle/working")
if os.path.exists("pygpt"):
    shutil.rmtree("pygpt")
if REPO_URL:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "pygpt"], check=True)
else:
    src = "/kaggle/input/pygpt-code"
    cand = [os.path.join(src, "pygpt"), src]
    for c in cand:
        if os.path.exists(os.path.join(c, "model.py")):
            shutil.copytree(c, "pygpt"); break
    else:
        raise SystemExit("set REPO_URL or add the 'pygpt-code' dataset as input")
print(sorted(os.listdir("pygpt")))
'''

# ----------------------------------------------------------------------------- 01
nb01 = nb([
    md('''
# 01 — Prepare data (CPU notebook)

**Settings (right sidebar):** Accelerator = **None**, Internet = **On**, Persistence = Files only.

Streams ~800M tokens of Python from `codeparrot/codeparrot-clean-train`, tokenizes with the
codeparrot 32k BPE tokenizer and writes `train.bin` / `val.bin` (uint16) to `/kaggle/working`.
Takes roughly 30–60 min. **Uses 0 GPU hours.**

When it finishes: **Save & Run All**, then on the finished version open the *Output* tab →
**New Dataset** → name it `pygpt-python-tokens`. Notebook 02 reads from that dataset.
'''),
    code(GET_CODE),
    code('''
%pip install -q datasets
import datasets, transformers; print("datasets", datasets.__version__, "transformers", transformers.__version__)
'''),
    code('''
# Quick test first (2M tokens, ~1 min). Set TRAIN_TOKENS to 800_000_000 for the real run.
TRAIN_TOKENS = 800_000_000
VAL_TOKENS   = 5_000_000
!cd /kaggle/working/pygpt && python data/prepare_data.py --out_dir /kaggle/working \\
    --train_tokens {TRAIN_TOKENS} --val_tokens {VAL_TOKENS}
'''),
    code('''
import json, os
print(json.dumps(json.load(open("/kaggle/working/meta.json")), indent=2))
for f in ("train.bin", "val.bin"):
    print(f, f"{os.path.getsize('/kaggle/working/'+f)/2**20:,.0f} MB")
print(open("/kaggle/working/sample.txt").read()[:1500])
'''),
    md('''
### Next
1. Confirm the sample above looks like real Python.
2. Output tab → **New Dataset** → `pygpt-python-tokens` (contains `train.bin`, `val.bin`, `meta.json`, `tokenizer/`).
3. Open Notebook 02.
'''),
])

# ----------------------------------------------------------------------------- 02
nb02 = nb([
    md('''
# 02 — Train (GPU T4 ×2 notebook)

**Settings:** Accelerator = **GPU T4 x2**, Internet = On, Input = dataset `pygpt-python-tokens` (+ `pygpt-code` if not cloning from GitHub).

Set `MODE` below and run all cells. Order on Day 1:

| MODE | What it does | Time |
|---|---|---|
| `sanity` | unit checks + 60-step smoke run on synthetic data (both GPUs) | ~3 min |
| `bench` | main config on real data for 60 steps → prints **tok/s** and **peak memory**; tests kill + resume | ~5 min |
| `main` | the 10-hour run — use **Save & Run All** so it survives closing the browser | 10 h |
| `resume` | continue a run from a previous version's `ckpt_latest.pt` (Appendix D of PROJECT.md) | — |

After `bench`, copy the printed tok/s into `MAX_ITERS` below before running `main`.
'''),
    code(GET_CODE),
    code('''
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count())
'''),
    code('''
# ================= SET THESE =================
MODE = "sanity"                 # "sanity" | "bench" | "main" | "resume"
N_GPUS = torch.cuda.device_count()
MICRO_BATCH = 16                # lower to 12 or 8 if bench shows OOM / > 15 GB peak
GRAD_ACCUM = 8
TIME_LIMIT_H = 10.0
# From bench: MAX_ITERS = (tok_per_sec * TIME_LIMIT_H * 3600) // (MICRO_BATCH * 512 * N_GPUS * GRAD_ACCUM)
MAX_ITERS = 3800
RESUME_PATH = ""                # e.g. "/kaggle/input/02-train/ckpt_latest.pt"
COMPILE = False
# =============================================
# data can arrive as the dataset "pygpt-python-tokens" or as Notebook 01's mounted output
import glob
DATA_DIR = "/kaggle/input/pygpt-python-tokens"
if MODE != "sanity" and not os.path.exists(DATA_DIR + "/train.bin"):
    hits = glob.glob("/kaggle/input/*/train.bin")
    assert hits, "add the pygpt-python-tokens dataset or Notebook 01's output as an input"
    DATA_DIR = os.path.dirname(hits[0])
print("DATA_DIR =", DATA_DIR)
TOK_PER_STEP = MICRO_BATCH * 512 * N_GPUS * GRAD_ACCUM
print(f"tokens/step = {TOK_PER_STEP:,}; {MAX_ITERS} steps = {MAX_ITERS*TOK_PER_STEP/1e6:.0f}M tokens")
os.chdir("/kaggle/working/pygpt")
'''),
    code('''
if MODE == "sanity":
    !python tests/sanity.py
    !python tests/make_synthetic_data.py --out_dir data/synthetic
    !torchrun --standalone --nproc_per_node={N_GPUS} train.py --config config/smoke.py \\
        --data_dir=data/synthetic --ckpt_dir=/kaggle/working/smoke --max_iters=60
    print("\\nsmoke run finished; loss should have dropped well below 10.4")
'''),
    code('''
if MODE == "bench":
    # 60 steps of the real config on real data: read tok/s and peak memory from the output
    !torchrun --standalone --nproc_per_node={N_GPUS} train.py --config config/gpt2_small_py.py \\
        --data_dir={DATA_DIR} --ckpt_dir=/kaggle/working/bench --micro_batch_size={MICRO_BATCH} \\
        --grad_accum_steps={GRAD_ACCUM} --max_iters=60 --eval_interval=30 --eval_iters=4 --warmup_iters=20 \\
        --compile={COMPILE}
    # resume test: continue the same run for 20 more steps from ckpt_latest.pt
    !torchrun --standalone --nproc_per_node={N_GPUS} train.py --config config/gpt2_small_py.py \\
        --data_dir={DATA_DIR} --ckpt_dir=/kaggle/working/bench --micro_batch_size={MICRO_BATCH} \\
        --grad_accum_steps={GRAD_ACCUM} --max_iters=80 --eval_interval=40 --eval_iters=4 --warmup_iters=20 \\
        --resume=/kaggle/working/bench/ckpt_latest.pt --compile={COMPILE}
    print("\\nIf both finished: note tok/s from the step lines and peak GPU memory from the [eval] lines.")
    print(f"MAX_ITERS = tok_per_sec * {TIME_LIMIT_H*3600:.0f} // {TOK_PER_STEP}")
'''),
    code('''
if MODE == "main":
    !torchrun --standalone --nproc_per_node={N_GPUS} train.py --config config/gpt2_small_py.py \\
        --data_dir={DATA_DIR} --ckpt_dir=/kaggle/working --micro_batch_size={MICRO_BATCH} \\
        --grad_accum_steps={GRAD_ACCUM} --max_iters={MAX_ITERS} --time_limit_hours={TIME_LIMIT_H} \\
        --compile={COMPILE}
'''),
    code('''
if MODE == "resume":
    assert RESUME_PATH, "set RESUME_PATH to the previous version's ckpt_latest.pt"
    !cp {RESUME_PATH} /kaggle/working/ckpt_latest.pt
    !torchrun --standalone --nproc_per_node={N_GPUS} train.py --config config/gpt2_small_py.py \\
        --data_dir={DATA_DIR} --ckpt_dir=/kaggle/working --micro_batch_size={MICRO_BATCH} \\
        --grad_accum_steps={GRAD_ACCUM} --max_iters={MAX_ITERS} --time_limit_hours={TIME_LIMIT_H} \\
        --resume=/kaggle/working/ckpt_latest.pt --compile={COMPILE}
'''),
    code('''
# Loss curve from log.csv (works after main / resume)
import pandas as pd, matplotlib.pyplot as plt, os
log = "/kaggle/working/log.csv"
if os.path.exists(log):
    df = pd.read_csv(log)
    tr = df[df.train_loss.notna()]; va = df[df.val_loss.notna()]
    plt.figure(figsize=(8,4))
    plt.plot(tr.tokens/1e6, tr.train_loss, label="train", alpha=.6)
    plt.plot(va.tokens/1e6, va.val_loss, "o-", label="val")
    plt.xlabel("tokens (M)"); plt.ylabel("loss (nats/token)"); plt.legend(); plt.grid(alpha=.3)
    plt.title("PyGPT training"); plt.savefig("/kaggle/working/loss_curve.png", dpi=150, bbox_inches="tight")
    print(va.tail(5))
'''),
    code('''
# Quick qualitative check after training
if os.path.exists("/kaggle/working/ckpt_best.pt"):
    !python generate.py --ckpt /kaggle/working/ckpt_best.pt --tokenizer {DATA_DIR}/tokenizer \\
        --prompt "def fibonacci(n):\\n    \\"\\"\\"Return the nth Fibonacci number.\\"\\"\\"\\n" --n 2 --temperature 0.5
'''),
    md('''
### Keeping the checkpoint
Outputs of a committed version (`ckpt_best.pt`, `ckpt_latest.pt`, `log.csv`, `loss_curve.png`) are kept by Kaggle.
To resume in a new version: *Add Input → Your Work → this notebook → previous version*, set `MODE="resume"` and
`RESUME_PATH="/kaggle/input/<slug>/ckpt_latest.pt"`.
'''),
])

# ----------------------------------------------------------------------------- 03
nb03 = nb([
    md('''
# 03 — Evaluate, export, demo (Day 2)

**Settings:** Accelerator = **GPU T4 x2** (CPU also works, ~10x slower eval), Internet = On.
**Inputs:** dataset `pygpt-python-tokens` **and** *Your Work → Notebook 02 (latest version)* so the
checkpoint mounts under `/kaggle/input/`.

Steps: `eval.py` (metrics + plots + samples, with the GPT-2 baseline) → `export.py` (fp16 ~220 MB)
→ zip a ready-to-upload HF Space bundle → optional: upload to HF Hub / launch a shareable Gradio link.
'''),
    code(GET_CODE),
    code('''
import glob, os
DATA_DIR = "/kaggle/input/pygpt-python-tokens"
if not os.path.exists(DATA_DIR + "/val.bin"):
    hits = glob.glob("/kaggle/input/*/val.bin")
    assert hits, "add the pygpt-python-tokens dataset or Notebook 01's output as an input"
    DATA_DIR = os.path.dirname(hits[0])
# find ckpt_best.pt from the Notebook 02 input (falls back to ckpt_latest.pt)
cands = sorted(glob.glob("/kaggle/input/*/ckpt_best.pt")) or sorted(glob.glob("/kaggle/input/*/ckpt_latest.pt"))
assert cands, "add Notebook 02's latest version as an input (Add Input -> Your Work)"
CKPT = cands[0]
LOG = os.path.join(os.path.dirname(CKPT), "log.csv")
print("ckpt:", CKPT)
os.chdir("/kaggle/working/pygpt")
'''),
    code('''
# E1-E5 with the vanilla GPT-2 baseline (~15-25 min on T4, mostly sampling)
!python eval.py --ckpt {CKPT} --data_dir {DATA_DIR} --log_csv {LOG} \\
    --out_dir /kaggle/working/results --gpt2
'''),
    code('''
import json
from IPython.display import Image, display, Markdown
print(json.dumps(json.load(open("/kaggle/working/results/metrics.json")), indent=2))
display(Image("/kaggle/working/results/loss_curve.png"))
display(Markdown(open("/kaggle/working/results/samples.md").read()[:3000]))
'''),
    code('''
# Export fp16 weights + tokenizer into app/, verify, and zip a complete HF Space bundle
!python export.py --ckpt {CKPT} --out app/model_fp16.pt --tokenizer_src {DATA_DIR}/tokenizer --verify
import shutil, os
os.makedirs("/kaggle/working/space", exist_ok=True)
for f in ("app/app.py", "app/requirements.txt", "app/model_fp16.pt", "model.py", "generate.py", "export.py"):
    shutil.copy(f, "/kaggle/working/space/" + os.path.basename(f))
shutil.copytree("app/tokenizer", "/kaggle/working/space/tokenizer", dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/space_bundle", "zip", "/kaggle/working/space")
print("wrote /kaggle/working/space_bundle.zip - upload its contents to a Gradio Space (or use the cell below)")
'''),
    code('''
# Optional: push weights + the Space in one go. Needs a write token in Kaggle Secrets as HF_TOKEN.
HF_USER = ""                       # e.g. "rehan"; leave empty to skip
if HF_USER:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import HfApi
    api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
    api.create_repo(f"{HF_USER}/pygpt", exist_ok=True)                                   # model repo
    api.upload_folder(folder_path="/kaggle/working/space", repo_id=f"{HF_USER}/pygpt")
    api.create_repo(f"{HF_USER}/pygpt-demo", exist_ok=True, repo_type="space", space_sdk="gradio")
    api.upload_folder(folder_path="/kaggle/working/space", repo_id=f"{HF_USER}/pygpt-demo", repo_type="space")
    print(f"https://huggingface.co/spaces/{HF_USER}/pygpt-demo")
'''),
    code('''
# Optional: temporary public Gradio link straight from Kaggle (fallback 2 in PROJECT.md 4.8)
# !cd /kaggle/working/pygpt && GRADIO_SHARE=1 python app/app.py
'''),
    md('''
### Done when
`results/metrics.json` numbers are on the results slide, the Space answers Prompt A in < 15 s,
and `space_bundle.zip` is downloaded as the laptop fallback.
'''),
])

for name, obj in (("01_prepare_data.ipynb", nb01), ("02_train.ipynb", nb02), ("03_eval_and_demo.ipynb", nb03)):
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", name)
