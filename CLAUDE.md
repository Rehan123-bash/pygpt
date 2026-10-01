# PyGPT — Claude Code project instructions

Read `PROJECT.md` for the full plan; this file is the short version plus current state.

## What this is
Course project (ML, Bennett University, solo, 3-day deadline Fri 2 Oct → Sun 4 Oct 2026).
Train a GPT-2-Small-architecture language model (12L/12H/768d, ctx 512, ~111M params with the
32k codeparrot tokenizer) **from scratch in PyTorch** on ~0.5–0.8B tokens of Python
(CodeParrot-clean), on **Kaggle 2× T4** (free, 30 GPU-h/week), and demo it live in class as a
**Gradio code-completion web app** (HF Spaces, laptop fallback). VS Code extension = stretch goal.

Graded on 4 criteria × 3 marks: existing systems & feasibility; objectives & methodology;
relevance of algorithms/techniques; design ↔ implementation sync. PROJECT.md §2 maps each.

## Hard rules
- Everything must **work live in class**. Prefer boring, tested choices over clever ones.
- Keep `PROJECT.md §6.2` (design ↔ implementation table) in sync with real file/function names.
- Never start a Kaggle GPU session without a specific task; budget is in PROJECT.md §9.
- No fine-tuning of pretrained models: the from-scratch training is the point (criterion 3).
- The AWS route is dead (quota denied twice). Kaggle only.

## Layout
- `model.py` — GPTConfig, CausalSelfAttention, MLP, Block, GPT (+ `configure_optimizers`, `generate`)
- `train.py` — loop: AdamW, warmup+cosine, fp16+GradScaler, grad-accum, DDP (torchrun), eval,
  `ckpt_latest.pt`/`ckpt_best.pt`, `log.csv`, `--resume`, `--time_limit_hours`. Config = file + `--key=value`.
- `data/prepare_data.py` — streams HF `codeparrot/codeparrot-clean-train` → `train.bin`/`val.bin` (uint16)
- `generate.py` — sample from a checkpoint (`complete()`, `trim_completion()`)
- `config/smoke.py` (tiny, CPU) · `config/gpt2_small_py.py` (main run)
- `tests/sanity.py` (6 checks) · `tests/make_synthetic_data.py` (fake learnable data)
- `eval.py` — E1–E5: `val_loss()`, `ast_validity()`, `functional_test()` (subprocess sandbox),
  loss-curve plot, samples.md; baselines random-init + `--gpt2`; `--fast` for CPU pipeline tests
- `export.py` — `export_fp16()` (drops tied lm_head, ~6× smaller), `load_exported()` (used by app), `--verify`
- `app/app.py` — Gradio UI; loads `app/model_fp16.pt` + `app/tokenizer/`; same file works locally,
  on HF Spaces (upload flat: app/* + model.py, generate.py, export.py) and Kaggle (`GRADIO_SHARE=1`)
- `notebooks/build_notebooks.py` → `01_prepare_data.ipynb` (CPU), `02_train.ipynb` (T4×2, `MODE`
  switch), `03_eval_and_demo.ipynb` (eval → export → Space bundle zip → optional HF upload)
- `slides/slides_outline.md` — all 13 slides drafted; ⟨…⟩ marks numbers to fill from real run
- To do: real Kaggle runs (Day 1), fill slides with real numbers + make PPTX, VS Code ext (stretch)

## Commands
```bash
pip install -r requirements.txt
python tests/sanity.py                                      # must print 6/6 before any long run
python tests/make_synthetic_data.py --out_dir data/synthetic
python train.py --config config/smoke.py                    # CPU smoke run, ~1 min
python train.py --config config/smoke.py --resume=runs/smoke/ckpt_latest.pt --max_iters=100
torchrun --standalone --nproc_per_node=2 train.py --config config/gpt2_small_py.py \
    --data_dir=/kaggle/input/pygpt-python-tokens --ckpt_dir=/kaggle/working --time_limit_hours=10
python generate.py --ckpt ckpt_best.pt --tokenizer <dir>/tokenizer --prompt "def fibonacci(n):\n"
python eval.py --ckpt ckpt_best.pt --data_dir <data> --gpt2   # full eval; --fast for a CPU pipeline test
python export.py --ckpt ckpt_best.pt --out app/model_fp16.pt --tokenizer_src <data>/tokenizer --verify
python app/app.py                                           # Gradio demo on :7860 (GRADIO_SHARE=1 for a link)
python notebooks/build_notebooks.py                         # regenerate .ipynb after editing them
```

## Verified so far (Day 0, CPU only)
sanity 6/6 · smoke run loss 10.4→4.1 · kill + `--resume` continues step/log/tokens · 2-process
`torchrun` (gloo) · time-limit exit saves ckpt · `prepare_data.py` logic against a faked stream ·
real codeparrot tokenizer downloaded: vocab 32768, **eos id 0** (ignore transformers' "eos_token_id
… got 50256" warning — stale Hub metadata, harmless) · **real HF streaming works**: `prepare_data.py`
pulled 2.1M train + 0.25M val tokens from both codeparrot-clean datasets into `data/tiny/`, decoded
sample is genuine Python · smoke config trained on that real data (10.4→7.0 @100 steps) · `eval.py
--fast` end-to-end on smoke ckpts, incl. offline tokenizer load from `<data>/tokenizer` (E1 ppl 854
vs random-init 33415; E2/E3 0% as expected at this scale) · `--gpt2` baseline sampler downloads and
samples (33% AST-valid spot check) · `export.py` 57→10 MB with `--verify` greedy-identical · Gradio
app serves HTTP 200 and completes on CPU (fp16 file loads as fp32 on CPU).
**Not yet tested:** T4 throughput/memory (Notebook 02 `MODE="bench"`), DDP on CUDA, full-size
800M-token prep on Kaggle, HF Space deploy, notebook 03 on Kaggle.

## Key numbers / decisions
- tokens/step = micro_batch(16) × 512 × 2 GPUs × grad_accum(8) = 131,072; `max_iters` set from
  measured tok/s: `(tok_s × 36000) // 131072`. Expect 10–15k tok/s per T4.
- LR 6e-4 → 6e-5 cosine, warmup 200, AdamW β=(0.9,0.95), wd 0.1 on matrices only, clip 1.0, fp16.
- Expected loss: 10.4 → ~4 @200 steps → ~2.5 @1000 → 1.6–1.9 at end. Val > 3.0 after 2 h = bug, stop.
- Checkpoint every 250 steps; only keep latest + best (~1.3 GB each) — Kaggle disk is ~20 GB.
- Tokenizer: `codeparrot/codeparrot` (vocab 32768, eos `<|endoftext|>`), saved to `<data>/tokenizer/`.
- Demo defaults: max_new_tokens 96, temperature 0.6, top-k 50, top-p 0.95.

## Day plan (details in PROJECT.md §8)
- **Day 1 (Fri):** Notebook 01 → dataset `pygpt-python-tokens`; Notebook 02 `sanity` → `bench`
  (read tok/s + peak mem, set `MICRO_BATCH`, `MAX_ITERS`) → `main` via **Save & Run All** by ~17:00.
- **Day 2 (Sat):** `eval.py` (val ppl, ast-validity %, 5-task functional test, baselines: random-init +
  vanilla GPT-2), `export.py` (fp16 weights-only ~220 MB), `app/app.py` (Gradio) → HF Space; laptop fallback; slides 1–13.
- **Day 3 (Sun):** polish, rehearse, buffer; VS Code extension only if everything else is green.

## Style
Plain PyTorch, no Lightning/HF Trainer. Short educational comments (the author presents this code).
Keep files small enough to show on a slide. Update this file's "Verified so far" as things get tested.
