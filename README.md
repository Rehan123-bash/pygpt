# PyGPT — a GPT-2 Small-class language model for Python, trained from scratch

Course project: train a decoder-only transformer on Python source code and ship it as a
code-completion web app. Full design, feasibility analysis, evaluation protocol and the
3-day plan are in [PROJECT.md](PROJECT.md).

## Layout

| Path | What |
|---|---|
| `model.py` | GPT architecture (attention, MLP, blocks, init, weight tying, sampling) |
| `train.py` | training loop: AdamW, warmup+cosine, fp16, grad accumulation, DDP, checkpoints, `--resume`, time limit |
| `data/prepare_data.py` | streams CodeParrot-clean from HF → `train.bin` / `val.bin` (uint16) |
| `generate.py` | sample completions from a checkpoint |
| `config/smoke.py`, `config/gpt2_small_py.py` | tiny debug config; main-run config |
| `tests/sanity.py` | unit checks: init loss, causal mask, weight tying, overfit-one-batch, generate |
| `tests/make_synthetic_data.py` | learnable fake data so the pipeline can be tested without downloads |
| `notebooks/01_prepare_data.ipynb` | Kaggle CPU notebook (data) |
| `notebooks/02_train.ipynb` | Kaggle 2×T4 notebook (sanity → bench → main run → resume) |
| `notebooks/03_eval_and_demo.ipynb` | Kaggle notebook: eval → export → HF Space bundle |
| `eval.py` | val perplexity, AST-validity %, 5-task functional test, baselines, plots |
| `export.py` | checkpoint → fp16 weights-only file (~220 MB) for the app |
| `app/app.py` | Gradio code-completion demo (local, HF Spaces, or Kaggle) |

## Quick start (any machine, CPU is fine)

```bash
pip install -r requirements.txt
python tests/sanity.py                                   # ~30 s
python tests/make_synthetic_data.py --out_dir data/synthetic
python train.py --config config/smoke.py                 # ~1 min on CPU; loss 10.4 → ~2
python generate.py --ckpt runs/smoke/ckpt_best.pt --prompt "def f(x):"
python eval.py --ckpt runs/smoke/ckpt_best.pt --data_dir data/synthetic --fast   # pipeline test
python export.py --ckpt runs/smoke/ckpt_best.pt --out app/model_fp16.pt --verify
python app/app.py                                        # Gradio demo at http://127.0.0.1:7860
```

## Real run (Kaggle)

1. Notebook 01 (CPU, Internet on): builds `train.bin`/`val.bin` → save output as dataset `pygpt-python-tokens`.
2. Notebook 02 (GPU T4×2): `MODE="sanity"` → `"bench"` (read tok/s, set `MAX_ITERS`) → `"main"` with **Save & Run All**.
3. Resume after a session cap: `MODE="resume"` with the previous version's `ckpt_latest.pt` as input.

```bash
# equivalent command line
torchrun --standalone --nproc_per_node=2 train.py --config config/gpt2_small_py.py \
    --data_dir=/kaggle/input/pygpt-python-tokens --ckpt_dir=/kaggle/working --time_limit_hours=10
```

Any config value can be overridden with `--key=value`.
