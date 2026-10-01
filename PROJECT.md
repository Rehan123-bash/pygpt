# PyGPT — Project Master Document

**Project:** Training a small GPT-style language model on Python source code to power a code-completion assistant
**Course:** Machine Learning course project, Bennett University
**Owner:** Rehan (solo assumed — see §14)
**Compute:** Kaggle free tier (2× T4)
**Deadline:** 3 days — Fri 2 Oct → Sun 4 Oct 2026
**Status:** Code for data → model → training → sampling written and tested on CPU (Day 0). Day 1 starts with Notebook 01 (data) and Notebook 02 `sanity` → `bench`.

---

## 0. TL;DR

Build a decoder-only transformer (GPT-2 Small architecture: 12 layers, 12 heads, 768-dim) **from scratch in PyTorch**, train it on ~0.5–0.8B tokens of permissively licensed Python (CodeParrot-clean) on Kaggle's free T4 GPUs in one ~10-hour run, evaluate it with held-out perplexity + a syntactic-validity test + a 5-task mini functional test (with baselines), and ship it as a **Gradio web app** on Hugging Face Spaces (local laptop fallback) that completes Python code live in class. A **VS Code extension** is the stretch goal once the web app works.

| Item | Value |
|---|---|
| Model | GPT-2 Small architecture, 12L / 12H / 768d, context 512 |
| Parameters | ~111M (124M-class — see §4.5 for why not exactly 124M) |
| Data | CodeParrot-clean (Python), ~800M-token subset prepared, train on what fits in ~10 h |
| Tokenizer | `codeparrot/codeparrot` BPE, vocab 32,768 |
| Hardware | Kaggle 2× T4 (16 GB each), fp16 mixed precision, DDP |
| Main training run | ~10 h, one background "Save & Run All" session, checkpoint every 250 steps |
| Target val loss | < 1.8 nats/token (~1.5 would be strong at this budget) |
| Demo | Gradio app: prompt → completion, temperature / top-k controls |
| Marks mapping | §2 |

---

## 1. End goal and scope

**Goal statement:** A working Python code-completion assistant powered by a language model we trained ourselves, with the full pipeline (data → tokenizer → model → training → evaluation → deployment) visible and reproducible.

**Definition of "done" for the class evaluation:**
1. Live demo: type a Python function signature/docstring into the web app, get a plausible completion in < 15 s.
2. A results slide with real numbers: loss curve, val perplexity, syntactic-validity %, mini functional-test pass rate, each compared against a baseline.
3. Slides that hit all four evaluation criteria (§2, §12).
4. GitHub repo whose structure matches the design document (§6).

**In scope:** everything above.
**Out of scope:** instruction tuning, multi-language support, RLHF, context > 512 tokens, beating CodeParrot/Copilot on quality.
**Stretch (only if Day 3 afternoon is free):** VS Code inline-completion extension backed by the same model (§13).

---

## 2. Evaluation criteria → how we score each

| Criterion (3 marks each) | What the evaluator wants to see | Where we deliver it |
|---|---|---|
| **Review of Existing Systems & Project Feasibility** | Knows the landscape (GPT-2, Codex, CodeParrot, StarCoder, Code Llama); can argue the project fits the time/compute available | §3 — comparison table, gap statement, compute-budget math, Kaggle constraints, AWS denial as a feasibility finding; slides 3–4 |
| **Objectives & Methodology** | Measurable objectives; a clear pipeline; a sound evaluation protocol | §4 — objectives list, pipeline diagram, data/tokenizer/model/training/eval sections; slides 5–6, 9 |
| **Relevance of Algorithms / Techniques** | Each technique is justified, not just used | §5 — technique → why table; slides 7–8 |
| **Synchronization of Project Design & Implementation** | The code matches the design; the demo runs the model you described | §6 — design ↔ file ↔ function ↔ verification map; repo tree; live demo; slides 10–11 |

---

## 3. Review of existing systems and feasibility

### 3.1 Existing systems

| System | Year | Params | Training data | Open weights? | Relevance to us |
|---|---|---|---|---|---|
| GPT-2 (Radford et al.) | 2019 | 124M–1.5B | WebText (English) | Yes | The architecture we reuse; not trained on code → our baseline |
| Codex / GitHub Copilot (Chen et al.) | 2021 | 12B | GitHub code | No | Defined the task and the HumanEval pass@k metric |
| CodeParrot (Hugging Face) | 2021 | 110M / 1.5B | ~50 GB Python (`codeparrot-clean`) | Yes | Closest prior work: same tokenizer and dataset family, ~50–100× more compute |
| The Stack → SantaCoder / StarCoder (BigCode) | 2022–23 | 1.1B / 15.5B | 80+ languages, permissive licenses | Yes | Data-licensing pipeline we follow in spirit |
| Code Llama (Meta) | 2023 | 7B–70B | 500B+ code tokens | Yes | State of the art for open code LMs |
| Tabnine, Amazon Q Developer, Cursor | — | undisclosed | — | No | Commercial IDE assistants; the UX we imitate |
| **PyGPT (ours)** | 2026 | ~111M | ~0.5–0.8B Python tokens | Yes | End-to-end reproducible on free GPUs; every stage visible and measured |

### 3.2 Gap / positioning

Every system above is either closed-source or trained with 2–4 orders of magnitude more compute than a student can access. None of them is useful for *learning* how a code LM is built. PyGPT's contribution is not SOTA quality; it is a complete, measured, reproducible pipeline at student scale, with honest numbers on what ~10 GPU-hours buys. The "small specialized model" angle (Python-only, 111M) is also a legitimate deployment story: it runs on a laptop CPU.

### 3.3 Feasibility

**Compute math (scaling-law estimate):**
- Training FLOPs ≈ 6 × N × D = 6 × 1.1e8 params × 5e8 tokens ≈ **3.3e17 FLOPs**
- T4 fp16 tensor-core peak = 65 TFLOPS; realistic model-FLOPs utilization for a 111M model at context 512 ≈ 20–30% → ~13–20 TFLOPS/GPU
- One T4: 3.3e17 / 1.5e13 ≈ 22,000 s ≈ **6 h** for 500M tokens. Two T4s with DDP (~1.8× scaling): **~3.5 h**.
- Expected throughput: 10–15k tokens/s per T4, 18–28k tokens/s on two. In a 10-hour run that is 0.65–1.0B tokens. We prepare ~800M tokens so we never exceed one epoch.
- Chinchilla-optimal for 111M params would be ~2.2B tokens; we are ~3–7× under-trained. Loss will still be decreasing when we stop. **This is expected and goes on the limitations slide.**

**Why from scratch and not fine-tuning:** criterion 3 rewards showing the architecture and training machinery. Fine-tuning GPT-2 would give better completions but hides most of the ML. Noted as a trade-off on the limitations slide.

**Compute sourcing:** AWS EC2 GPU quota request was denied twice (new account, no usage history; case 179063147200411). Kaggle's free tier (30 GPU-h/week) is sufficient for this budget. This is itself a feasibility finding worth one line on slide 4.

**Kaggle constraints and how we handle them:**

| Constraint | Value | Handling |
|---|---|---|
| Weekly GPU quota | 30 h | Budget in §9; data prep runs in a **CPU-only** session (0 GPU-h) |
| Max session length | 12 h (verify in the session panel) | Main run capped at ~10 h; checkpoint every 250 steps; resume procedure in Appendix D |
| GPU choices | 1× P100 or 2× T4 | **2× T4** — tensor cores for fp16, and two GPUs |
| Disk `/kaggle/working` | ~20 GB | Keep only `ckpt_latest.pt` + `ckpt_best.pt` (~1.3 GB each) + logs |
| RAM | ~30 GB | `np.memmap` the token files; never load fully |
| Internet | Off by default; needs phone-verified account | Turn on for the data-prep and train notebooks (HF downloads, HF Hub upload) |
| Interactive session state | `/kaggle/working` is lost when an interactive session ends | Enable "Persistence: Files only" for dev; run the main training via **Save & Run All (Commit)**, which runs in the background and saves outputs as a notebook version |
| Browser disconnects | Interactive sessions can die | Same answer: Commit for anything > 30 min |

**Time feasibility:** 3 days is enough for exactly one full training run plus one resume. The plan (§8) front-loads everything that could break (data, smoke test, checkpoint/resume) into Day 1 before launching the run.

### 3.4 Key references

1. Vaswani et al., 2017 — Attention Is All You Need
2. Radford et al., 2019 — Language Models are Unsupervised Multitask Learners (GPT-2)
3. Chen et al., 2021 — Evaluating Large Language Models Trained on Code (Codex, HumanEval, pass@k)
4. Tunstall, von Werra, Wolf, 2022 — Natural Language Processing with Transformers, ch. 10 (CodeParrot)
5. Kocetkov et al., 2022 — The Stack; Li et al., 2023 — StarCoder
6. Rozière et al., 2023 — Code Llama
7. Kaplan et al., 2020 — Scaling Laws for Neural Language Models; Hoffmann et al., 2022 — Training Compute-Optimal LLMs (Chinchilla)
8. Sennrich et al., 2016 — Neural Machine Translation of Rare Words with Subword Units (BPE)
9. Loshchilov & Hutter, 2019 — Decoupled Weight Decay Regularization (AdamW)
10. Micikevicius et al., 2018 — Mixed Precision Training
11. Press & Wolf, 2017 — Using the Output Embedding to Improve Language Models (weight tying)
12. Holtzman et al., 2020 — The Curious Case of Neural Text Degeneration (nucleus sampling)
13. Karpathy — nanoGPT (reference implementation pattern)

---

## 4. Objectives and methodology

### 4.1 Objectives (measurable)

| # | Objective | Success measure |
|---|---|---|
| O1 | Build a GPT-2-Small-class decoder-only transformer from scratch in PyTorch | Init loss ≈ ln(32768) = 10.4; overfits one batch to ~0; passes causal-mask test |
| O2 | Prepare a clean Python pretraining corpus | ≥ 800M tokens in `train.bin`, ~5M in `val.bin`, decode round-trip verified |
| O3 | Train to a useful loss on free GPUs within budget | Val loss < 1.8 in ≤ 10 GPU-h on 2× T4; loss curve logged |
| O4 | Evaluate quantitatively against baselines | Val perplexity; syntactic-validity % (ours vs random-init vs vanilla GPT-2); mini functional pass@k |
| O5 | Deploy as a code-completion web app | HF Space live; completion for a 30-token prompt in < 15 s on CPU |
| O6 | Keep design and implementation in sync | §6 table filled with real file/function names; repo public |
| O7 (stretch) | IDE integration | VS Code inline completions working in a 60-s video |

### 4.2 Pipeline

```
Hugging Face Hub  (codeparrot-clean-train, streamed)
        │   light filters: < 100k chars, not auto-generated, mostly alphabetic
        ▼
data/prepare_data.py ── BPE tokenize (codeparrot 32k) ── <|endoftext|> between files
        │
        ▼
train.bin / val.bin   (uint16 memmap)  ──►  saved as a Kaggle Dataset
        │
        ▼
train.py   (model.py GPT · AdamW · warmup+cosine · fp16+GradScaler · DDP 2×T4 · checkpoints · CSV log)
        │
        ▼
ckpt_best.pt ──► eval.py  (val loss/ppl · ast-validity % · mini functional test · baselines · plots)
        │
        ▼
export.py ──► weights fp16 (~220 MB) ──► app/app.py (Gradio) ──► HF Spaces | laptop | Kaggle share link
        │
        └──(stretch) server.py FastAPI /complete ──► vscode-ext/ inline completions
```

### 4.3 Data

- **Source:** `codeparrot/codeparrot-clean-train` (Python files from GitHub, already near-deduplicated, auto-generated files removed, line-length filtered). Validation: `codeparrot/codeparrot-clean-valid` (disjoint files).
- **Fallback if unavailable:** `codeparrot/github-code-clean` filtered to `languages=["Python"]`, or `bigcode/the-stack-dedup` `data/python` (gated; needs HF login).
- **Subset:** stream and tokenize until **~800M train tokens** (~1.6 GB as uint16) and **~5M val tokens**. Streaming avoids downloading the ~50 GB full set.
- **Filters (light, since the set is pre-cleaned):** skip files > 100k chars; skip files whose mean line length > 100 chars (minified/data dumps); skip if < 25% alphabetic characters; skip if "auto-generated" / "do not edit" appears in the first 5 lines.
- **Format:** each file tokenized, `<|endoftext|>` appended, concatenated into one stream; written incrementally to a `np.uint16` file (vocab 32,768 fits exactly). Training samples random 512-token windows.
- **Licensing note for the slides:** CodeParrot-clean is derived from public GitHub; we use it for a non-commercial course project. BigCode's permissive-license filtering is the production-grade answer.

### 4.4 Tokenizer

- **Choice:** reuse `AutoTokenizer.from_pretrained("codeparrot/codeparrot")` — a GPT-2-style byte-level BPE trained on Python, vocab 32,768.
- **Why not GPT-2's tokenizer:** GPT-2's English vocab encodes 4-space indentation as multiple tokens and splits common Python identifiers badly; the code-trained vocab compresses Python ~30–40% better, so every training token carries more signal.
- **Why not train our own:** ~1–2 h of extra work and risk for no demo gain. Listed as future work; the slides explain BPE either way.
- **Deployment:** `tokenizer.save_pretrained("app/tokenizer")` is committed to the repo so the app works offline.

### 4.5 Model architecture

| Hyperparameter | Value | Why |
|---|---|---|
| n_layer / n_head / n_embd | 12 / 12 / 768 | GPT-2 Small; the architecture pitched in the project |
| block_size (context) | 512 | ~1.4× faster per token than 1024 on T4 and halves activation memory; enough for function-level completion |
| vocab_size | 32,768 | codeparrot tokenizer |
| dropout | 0.0 | Single-epoch pretraining with data ≫ params → no overfitting; dropout only slows convergence |
| bias | False | Marginally faster; matches modern practice |
| Weight tying | Yes (`tok_emb.weight` = `lm_head.weight`) | Saves 25M params; improves small-model efficiency |
| Init | N(0, 0.02); residual-projection weights scaled by 1/√(2·n_layer) | GPT-2 init; keeps the residual stream variance bounded across 24 residual adds |
| Norm | Pre-LayerNorm | More stable than post-LN at 12 layers |
| Activation | GELU | GPT-2 |
| Attention | `F.scaled_dot_product_attention(..., is_causal=True)` | Uses the memory-efficient kernel on T4 (the flash kernel needs Ampere+) |

**Parameter count (with 32k vocab, tied embeddings):**
- Token embedding 32,768 × 768 = 25.2M
- Position embedding 512 × 768 = 0.4M
- Per block: QKV 768×2304 (1.77M) + proj 768×768 (0.59M) + MLP 768×3072×2 (4.72M) ≈ 7.1M; × 12 = 85.0M
- **Total ≈ 111M.** The same architecture with GPT-2's 50,257-token vocab and 1024 positions is the familiar 124M. Say "GPT-2 Small architecture (124M-class)" on slides and show this breakdown — it demonstrates understanding rather than hiding the difference.

### 4.6 Training recipe

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW, β = (0.9, 0.95), ε = 1e-8 | Weight decay 0.1 on 2-D weight matrices only; 0 on biases, norms, embeddings |
| Peak LR | 6e-4 | GPT-2 Small standard |
| Schedule | Linear warmup 200 steps → cosine decay to 6e-5 at `max_iters` | |
| Tokens per optimizer step | ~131k = micro-batch 16 × 512 ctx × 2 GPUs × grad-accum 8 | Micro-batch tuned in the smoke test (fits 16 GB?) |
| `max_iters` | `(measured tok/s × 36,000 s) / tokens_per_step` | ≈ 3,800 at 500M tokens; set after the smoke test |
| Precision | fp16 autocast + `GradScaler` | T4 has no bf16; scaler prevents gradient underflow |
| Gradient clipping | 1.0 | Prevents fp16 loss spikes |
| Multi-GPU | `torchrun --standalone --nproc_per_node=2 train.py` (DDP) | Fallback: single GPU if DDP misbehaves in the smoke test |
| Eval | Every 250 steps on 40 val batches | Logged with train loss, LR, tok/s to `log.csv` |
| Checkpoints | Every 250 steps → `ckpt_latest.pt` (model + optimizer + scaler + step + RNG, ~1.3 GB); `ckpt_best.pt` on new best val | Both overwritten in place to respect disk |
| Resume | `--resume PATH` | Tested in the smoke test before the main run — non-negotiable |
| Data sampling | Random windows from the memmap; each DDP rank seeded by `seed + rank` | |
| `torch.compile` | Try in smoke test; disable if it errors on T4 (sm75) | Worth ~10–20% if it works |
| Logging | CSV every 10 steps; stdout every 50 | Plotted by `eval.py` |

**Expected loss trajectory (for sanity checks during the run):** step 0 ≈ 10.4 → ~4 by step 200 → ~2.5 by step 1,000 → ~1.6–1.9 by the end. If val loss is > 3.0 after 2 hours, something is wrong (LR, data, mask); stop and debug rather than burn hours.

### 4.7 Evaluation protocol

| # | Metric | How | Baselines |
|---|---|---|---|
| E1 | Val loss and perplexity (e^loss) | ~5M held-out tokens from `codeparrot-clean-valid` | Random-init model (≈ 32,768 ppl); vanilla GPT-2 124M (English-only; note different tokenizer → compare loss per character, or just report its ast-validity) |
| E2 | Syntactic validity % | 20 prompts × 5 samples (T = 0.6, top-k 50, 96 tokens); trim to last complete line; `ast.parse(prompt + completion)` | Random-init (~0%), vanilla GPT-2, ours |
| E3 | Mini functional test ("HumanEval-lite") | 5 tasks (fibonacci, is_prime, reverse_string, factorial, max_of_list), 3 hidden unit tests each, 10 samples per task, sandboxed `exec` with 2-s timeout; report pass@1 and pass@10 | Vanilla GPT-2; ours. Honest expectation: 1–3 of 5 tasks at pass@10 |
| E4 | Loss curves | Train/val loss vs tokens and vs wall-clock, from `log.csv` | — |
| E5 | Qualitative samples | 6 curated prompts (Appendix B), including one failure case shown honestly | — |

All numbers go into `results/metrics.json` and the results slide.

### 4.8 Deployment (Gradio)

- **UI:** code textbox (prompt) → "Complete" button → code textbox (prompt + completion, completion highlighted). Sliders: `max_new_tokens` (32–256, default 96), `temperature` (0.1–1.2, default 0.6), `top_k` (default 50), `top_p` (default 0.95). Three example prompts as one-click buttons.
- **Stopping:** stop at `<|endoftext|>`, or when a new column-0 `def`/`class`/`@` line starts after the function body began, or at `max_new_tokens`.
- **Model loading:** `model.py` + fp16 weights-only file (~220 MB) + `app/tokenizer/`. No HF conversion; fewer moving parts.
- **Hosting (primary):** Hugging Face Space, Gradio SDK, free CPU tier. ~8–15 tok/s on CPU → 96 tokens in ~7–12 s. Acceptable for a demo; set default tokens to 96.
- **Fallback 1:** run `python app/app.py` on the laptop (CPU is fine for 111M). Verified on Day 2.
- **Fallback 2:** Kaggle notebook with GPU + `demo.launch(share=True)` (temporary public link).
- **Fallback 3:** 60-second screen recording of the working demo, embedded in the slides.

---

## 5. Algorithms and techniques — why each one

| Technique | Where | Why it is the right choice here |
|---|---|---|
| Byte-level BPE (Sennrich 2016; GPT-2) | tokenizer | Open-vocabulary subword units; code-trained vocab makes indentation and identifiers cheap |
| Decoder-only transformer with causal self-attention (Vaswani 2017; Radford 2019) | `model.py` | Next-token prediction *is* code completion; fully parallel training over the sequence |
| Pre-LayerNorm, GELU, residual connections | `model.py` | Stable optimization at 12 layers without warmup tricks |
| Learned positional embeddings | `model.py` | Simple, matches GPT-2; fine for fixed 512 context |
| Weight tying (Press & Wolf 2017) | `model.py` | 25M fewer params; better sample efficiency for small models |
| GPT-2 scaled residual init | `model.py` | Keeps activation variance from growing with depth |
| Memory-efficient scaled-dot-product attention | `model.py` | Linear attention memory → larger micro-batch on 16 GB |
| Cross-entropy next-token loss | `model.py` | Maximum-likelihood objective; perplexity falls out directly |
| AdamW (Loshchilov & Hutter 2019) | `train.py` | Decoupled weight decay regularizes without fighting Adam's scaling |
| Linear warmup + cosine decay | `train.py` | Warmup avoids early divergence with Adam; low final LR sharpens the minimum |
| fp16 mixed precision + dynamic loss scaling (Micikevicius 2018) | `train.py` | 2–3× throughput on T4 tensor cores; scaler prevents underflow |
| Gradient accumulation | `train.py` | Large effective batch within 16 GB |
| Gradient clipping | `train.py` | Caps rare fp16 spikes |
| DistributedDataParallel | `train.py` | Uses both Kaggle T4s with near-linear scaling |
| Checkpoint / resume | `train.py` | Survives Kaggle's session cap |
| Temperature, top-k, nucleus sampling (Holtzman 2020) | `model.generate` | Trades determinism vs diversity; demoed live |
| Scaling laws (Kaplan 2020; Hoffmann 2022) | §3.3 | Sizes the run and sets honest expectations |
| Perplexity, syntactic validity, pass@k (Chen 2021) | `eval.py` | Intrinsic + functional evaluation with baselines |

---

## 6. Design ↔ implementation map (criterion 4)

### 6.1 Repository layout

```
pygpt/
├── PROJECT.md                 ← this document
├── README.md                  ← 10-line "how to run"
├── requirements.txt
├── config/
│   ├── smoke.py               ← tiny config (4L/4H/128d, ctx 128) for bug-finding
│   └── gpt2_small_py.py       ← main-run config (Appendix A)
├── data/
│   └── prepare_data.py        ← HF stream → filter → tokenize → train.bin / val.bin
├── model.py                   ← GPTConfig, CausalSelfAttention, MLP, Block, GPT (+ generate)
├── train.py                   ← loop, AdamW, LR schedule, fp16, DDP, checkpoints, CSV log
├── generate.py                ← CLI sampling from a checkpoint
├── tests/
│   ├── sanity.py              ← 6 unit checks (run before any long training)
│   └── make_synthetic_data.py ← learnable fake data for pipeline tests without downloads
├── eval.py                    ← val loss/ppl, ast-validity, mini functional test, baselines, plots
├── export.py                  ← ckpt → fp16 weights-only file for the app
├── app/
│   ├── app.py                 ← Gradio UI
│   ├── tokenizer/             ← saved codeparrot tokenizer (offline)
│   └── requirements.txt
├── notebooks/
│   ├── build_notebooks.py     ← generates the .ipynb files below (keeps them in sync with the code)
│   ├── 01_prepare_data.ipynb  ← Kaggle CPU notebook
│   ├── 02_train.ipynb         ← Kaggle GPU notebook: MODE = sanity / bench / main / resume
│   └── 03_eval_and_demo.ipynb ← eval → export → HF Space bundle / upload
├── results/
│   ├── log.csv · loss_curve.png · metrics.json · samples.md
├── slides/
└── (stretch) server.py · vscode-ext/
```

### 6.2 Component map

| Design component | File | Key symbol | Verified by |
|---|---|---|---|
| Data filtering & tokenization (§4.3) | `data/prepare_data.py` | `keep_file()`, `tokenize_stream()` | Token count printed and re-counted from disk; one `<|endoftext|>` per file; decoded window in `sample.txt` |
| Causal self-attention (§4.5) | `model.py` | `CausalSelfAttention` | `tests/sanity.py` check 2: changing token t+1 never changes logits at ≤ t |
| Transformer block / full model | `model.py` | `Block`, `GPT`, `GPT._init_weights` | `tests/sanity.py` checks 1, 3, 4, 6: init loss ≈ 10.4, weight tying, overfit-one-batch < 0.1, ~111M params |
| Optimizer with decayed / undecayed groups (§4.6) | `model.py` | `GPT.configure_optimizers()` | used by `train.py`; smoke-run loss falls 10.4 → 4.1 in 100 steps |
| LR schedule, precision, DDP, eval, checkpoints, time limit (§4.6) | `train.py` | `get_lr()`, `estimate_loss()`, `save_ckpt()`, `should_stop_for_time()` | `config/smoke.py` run on synthetic data; kill-and-`--resume` test (step counter, log and token count continue); 2-process `torchrun` run; time-limit exit saves `ckpt_latest.pt` |
| Sampling (§4.8) | `model.py`, `generate.py` | `GPT.generate()`, `complete()`, `trim_completion()` | `tests/sanity.py` check 5; samples in `results/samples.md` |
| Evaluation (§4.7) | `eval.py` | `val_loss()`, `ast_validity()`, `functional_test()` | `--fast` run on the smoke ckpt: E1 ppl 120 vs random-init 33,909; sandboxed `exec` via subprocess; `results/metrics.json` written |
| Export | `export.py` | `export_fp16()`, `load_exported()` | `--verify`: greedy 20-token continuation identical before/after fp16 round-trip |
| Demo (§4.8) | `app/app.py` | `complete_code()` (wraps `generate.complete()`) | serves HTTP 200 locally; completes on CPU; HF Space on Day 2 |
| (stretch) IDE integration (§13) | `server.py`, `vscode-ext/` | `/complete`, `InlineCompletionItemProvider` | 60-s video |

**Status (end of Day 0):** every row above is written and verified on CPU (sanity 6/6, smoke run, resume, 2-process DDP, time-limit exit, data-prep logic against a faked stream; eval/export/app exercised end-to-end against the smoke checkpoint, with real results pending the trained model). Notebook 03 and `slides/slides_outline.md` are drafted. Not verifiable offline: the real HF dataset download and T4 throughput (Notebook 02 `bench`, Day 1), the GPT-2 baseline download, and the HF Space deploy (Day 2).

Rule for the presentation: every box on the architecture slide names the file and class that implements it.

---

## 7. Kaggle workflow

**Notebook 01 — data prep (Accelerator: None, Internet: On, Persistence: Files only)**
1. `pip install -q datasets` (if not preinstalled); `transformers` is preinstalled.
2. Run `prepare_data.py` → `/kaggle/working/train.bin`, `val.bin`, `meta.json` (token counts, tokenizer name).
3. Save & Run All → when done, **Output tab → "New Dataset"** → name `pygpt-python-tokens`. This makes the tokens reusable across notebooks and versions. (~30–40 min CPU, 0 GPU-h.)

**Notebook 02 — training (Accelerator: GPU T4 ×2, Internet: On)**
1. Add Input → Datasets → `pygpt-python-tokens` → files at `/kaggle/input/pygpt-python-tokens/`.
2. Cells: `%%writefile model.py`, `%%writefile train.py`, `%%writefile config/...` (or `!git clone` the repo).
3. Smoke test (interactive): `!torchrun --standalone --nproc_per_node=2 train.py --config smoke` (5 min), then main config for 300 steps: record tok/s and peak memory (`torch.cuda.max_memory_allocated`), then kill and `--resume` to prove it works.
4. Set `max_iters` from measured tok/s. Set `--time_limit_hours 10` in `train.py` so it saves and exits cleanly before the session cap.
5. **Save & Run All (Commit)** for the main run. Close the browser; it keeps running.
6. Outputs (`ckpt_latest.pt`, `ckpt_best.pt`, `log.csv`) appear under the notebook version's Output.

**Notebook 03 — eval & demo (GPU T4 ×2 for speed; CPU works too)**
1. Add Input → Your Work → Notebook 02 (latest version) → `/kaggle/input/02-train/ckpt_best.pt`.
2. Run `eval.py`; run `export.py`; test `app.py` with `share=True`; upload weights to HF Hub (`huggingface_hub`, token in Kaggle Secrets).

Resume procedure: Appendix D.

---

## 8. Three-day plan

### Day 1 — Fri 2 Oct (national holiday → full day available)

| Time (IST) | Task | GPU-h |
|---|---|---|
| 09:00–10:30 | Create GitHub repo + Kaggle notebook 01. Write `prepare_data.py`. Start streaming/tokenizing (~800M tokens). Save as Kaggle Dataset when done. | 0 |
| 10:30–13:00 | Write `model.py`, `train.py`, both configs. Unit sanity in a CPU/GPU session: forward shapes, init loss ≈ 10.4, causal-mask test, overfit one batch. | 0.5 |
| 14:00–16:00 | Smoke test on 2× T4: `smoke` config 5 min → main config 300 steps. Record tok/s, memory. **Test kill + `--resume`.** Try `torch.compile`; keep or drop. Decide micro-batch, `max_iters`. If DDP misbehaves > 20 min, go single-GPU and move on. | 1.5 |
| 16:00–17:00 | Final config. Write `generate.py`. Commit everything to GitHub. | 0 |
| **17:00** | **Launch main run via Save & Run All** (`--time_limit_hours 10` → finishes ~03:00). | 10 |
| Evening | Draft slides 1–6 (title, problem, existing systems, feasibility, objectives, methodology). | 0 |

### Day 2 — Sat 3 Oct

| Time (IST) | Task | GPU-h |
|---|---|---|
| 09:00–09:30 | Check Notebook 02 output. Pull `log.csv`, plot loss. If the run died early, start a resume version now (runs in background while you work). | ≤ 6 |
| 09:30–12:00 | Write `eval.py`: E1–E5 with baselines. Produce `metrics.json`, `loss_curve.png`, `samples.md`. | 0.5 |
| 13:00–16:00 | Write `export.py` and `app/app.py`. Test in Kaggle. Upload fp16 weights to HF Hub. Create the HF Space; verify from phone and laptop. | 0.3 |
| 16:00–18:00 | Laptop fallback: clone repo, `python app/app.py`, confirm CPU latency. Record a 60-s demo video. | 0 |
| Evening | Slides 7–13 (architecture/techniques, data, results, demo, design↔implementation, limitations, references). Fill §6.2 with final names. | 0 |

### Day 3 — Sun 4 Oct

| Time (IST) | Task |
|---|---|
| Morning | Polish slides; architecture diagram with file names; rehearse twice with a timer (target 7–8 min talk + 2 min demo). Finalize `README.md`. |
| 12:00–14:00 | Buffer for anything broken. Re-run E2/E3 on the final checkpoint if a resume finished overnight. |
| 14:00 onward | **Only if everything above is green:** VS Code extension stretch (§13). Otherwise rest. |
| Evening | Presentation-day checklist (Appendix C). |

---

## 9. GPU-hour budget (Kaggle: 30 h/week)

| Activity | GPU-h |
|---|---|
| Data prep | 0 (CPU session) |
| Unit sanity + smoke test + resume test | 2 |
| Main run | 10 |
| Resume run (only if needed) | ≤ 6 |
| Eval + export + demo test | 1 |
| **Committed** | **≤ 19** |
| Reserve (bugs, second attempt) | ~11 |

Rule: never start a GPU session without a specific task; stop the session the moment the task is done.

---

## 10. Class demo script (2 minutes)

1. Open the Space (already loaded, warmed up with one request before the talk).
2. Prompt A (click example): `def fibonacci(n):` + docstring → Complete. Read the output aloud; point out it uses recursion/loop correctly (or doesn't — say so).
3. Prompt B: a NumPy function signature → shows it learned library idioms.
4. Prompt C: `class Stack:` with `__init__` → shows multi-method structure.
5. Same prompt, temperature 0.2 vs 0.9 → explain sampling in one sentence.
6. One honest failure (Prompt F) → "this is what 0.5B tokens and 10 GPU-hours buys; CodeParrot used ~50–100× more."
7. Back to slides for results + design↔implementation.

Test all six prompts the night before and keep the known-good outputs as screenshots.

---

## 11. Risks and fallbacks

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Kaggle session dies mid-run | Medium | Checkpoint every 250 steps; `--time_limit_hours`; resume procedure (App. D); resume budget reserved |
| DDP flaky in Kaggle notebook | Medium | 20-min cap on debugging; fall back to single T4 (halves throughput; still ~350–500M tokens in 10 h) |
| fp16 NaN / loss spike | Low–medium | `GradScaler`, clip 1.0; if NaN, resume from last ckpt with LR × 0.5 |
| Val loss not falling (bug) | Low | Overfit-one-batch and causal-mask tests on Day 1 catch almost all of these before the long run; 2-hour sanity threshold (§4.6) |
| HF dataset renamed/removed | Low | Fallbacks listed in §4.3 |
| Out of GPU quota | Low | Budget in §9; data prep on CPU; no idle sessions |
| Completions look bad | Medium | Expected to be "plausible, not correct". Curate 6 prompts; show a failure honestly; emphasize measured results over cherry-picked quality |
| HF Space down / no internet in class | Medium | Laptop fallback; Kaggle share link; recorded video; screenshots |
| Running out of time for the extension | High | It is a stretch goal; "future work" slide is already written to cover it |
| Evaluator asks "why not fine-tune?" | Certain | Answer prepared in §3.3 |

---

## 12. Slides outline (13 slides)

| # | Slide | Criterion |
|---|---|---|
| 1 | Title, name, course | — |
| 2 | Problem and end goal: a Python completion assistant; what "working" means | — |
| 3 | Existing systems table + gap statement | C1 |
| 4 | Feasibility: compute math, Kaggle budget, AWS denial → Kaggle | C1 |
| 5 | Objectives O1–O7 with success measures | C2 |
| 6 | Methodology pipeline diagram | C2 |
| 7 | Architecture (block diagram with file/class names) + param breakdown | C3, C4 |
| 8 | Data, tokenizer, training recipe — each row with a "why" | C3 |
| 9 | Results: loss curve; metrics table with baselines; 3 samples | C2, C3 |
| 10 | Live demo | C4 |
| 11 | Design ↔ implementation map (repo tree + §6.2 table) | C4 |
| 12 | Limitations and future work (under-trained vs Chinchilla; own tokenizer; longer context; instruction tuning; VS Code extension) | C1 |
| 13 | References | C1 |

---

## 13. Stretch: VS Code extension

Only start this on Day 3 afternoon if the web app and slides are done.

- **Backend:** `server.py` — FastAPI with `POST /complete {prompt, max_new_tokens, temperature}` → `{completion}`. Reuses `app.complete()`. Run locally (`uvicorn server:app`) or in Kaggle behind `ngrok`.
- **Extension:** scaffold with `npx yo code` (TypeScript). Register `vscode.languages.registerInlineCompletionItemProvider({ language: "python" }, provider)`. On trigger, send the previous ~40 lines as the prompt, debounce 300 ms, return one `InlineCompletionItem`. ~120 lines of TypeScript.
- **Demo:** 60-s screen recording of ghost-text completions in VS Code; run from the laptop with the local server.
- **Time:** ~4 h including debugging. If not done, it lives on the future-work slide, which is already written.

---

## 14. Decisions log and open questions

**Decisions made (change if you disagree):**

| # | Decision | Reason |
|---|---|---|
| D1 | Train from scratch, not fine-tune | Criterion 3; shows the full ML pipeline |
| D2 | GPT-2 Small architecture (12/12/768), context 512 | Pitched architecture; 512 for T4 speed/memory |
| D3 | Reuse the codeparrot 32k tokenizer instead of training one | Saves ~2 h; better code compression than GPT-2's vocab |
| D4 | CodeParrot-clean streamed subset, ~800M tokens prepared | Avoids a 50 GB download; never exceeds one epoch |
| D5 | Kaggle 2× T4, fp16, DDP; single-GPU fallback | Tensor cores + two GPUs beat one P100 |
| D6 | Gradio on HF Spaces first; VS Code extension as stretch | Per your answer; the app's `complete()` becomes the extension's backend |
| D7 | Deliverables assumed: live demo + slides (+ this document as the written design) | You had no preference; a formal report is not planned |
| D8 | Solo project assumed | You had no preference |

**Open questions (answer whenever; none block Day 1):**
1. Presentation date and time? (Determines whether Day 3 is really available.)
2. Is a written report required by the course, or are slides + demo enough?
3. Any course rule against using a pretrained tokenizer or a public dataset?
4. Team size — if there are 2+ people, Day 2 splits cleanly into (eval + slides) and (app + hosting).
5. Do you want the GitHub repo public from Day 1 (recommended for the design↔implementation criterion)?

---

## Appendix A — Main-run config (`config/gpt2_small_py.py`)

```python
# model
n_layer, n_head, n_embd = 12, 12, 768
block_size = 512
vocab_size = 32768
dropout = 0.0
bias = False

# data
data_dir = "/kaggle/input/pygpt-python-tokens"
micro_batch_size = 16          # set from smoke test (try 16; fall back to 12/8)
grad_accum_steps = 8           # per GPU; tokens/step = 16*512*2*8 = 131,072

# optimizer / schedule
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 200
max_iters = 3800               # set from measured tok/s: (tok_s * 36000) // 131072
beta1, beta2 = 0.9, 0.95
weight_decay = 0.1
grad_clip = 1.0

# system
dtype = "float16"              # T4: no bf16
compile = False                # flip to True if the smoke test passes with it
eval_interval = 250
eval_iters = 40
log_interval = 10
ckpt_dir = "/kaggle/working"
time_limit_hours = 10.0        # save + exit cleanly before the session cap
seed = 1337
```

## Appendix B — Demo prompts (test all before class)

```python
# A
def fibonacci(n):
    """Return the nth Fibonacci number."""

# B
import numpy as np

def normalize(arr):
    """Scale arr to zero mean and unit variance."""

# C
class Stack:
    def __init__(self):

# D
def is_prime(n):

# E
def read_json(path):
    """Load a JSON file and return the parsed object."""

# F (expected failure / honesty slide)
def solve_sudoku(board):
```

## Appendix C — Presentation-day checklist

- [ ] HF Space loads and completes Prompt A in < 15 s (warm it up 5 min before)
- [ ] Laptop fallback: `python app/app.py` works offline (weights + tokenizer local)
- [ ] Demo video (60 s) and screenshots of all six prompts in the slides folder
- [ ] Slides exported to PDF as well as PPTX
- [ ] `results/metrics.json` numbers match the results slide
- [ ] GitHub repo public, README runs, §6.2 table matches actual file/function names
- [ ] One-line answers ready for: "why not fine-tune?", "why 512 context?", "why is it 111M not 124M?", "what would you do with 10× compute?"
- [ ] Phone hotspot as backup internet

## Appendix D — Resuming a Kaggle run

1. Open Notebook 02 → **Add Input → Your Work** → select Notebook 02's latest version → its outputs mount at `/kaggle/input/<notebook-slug>/`.
2. In the first cell set `RESUME = "/kaggle/input/<notebook-slug>/ckpt_latest.pt"` and keep `max_iters` the same (the schedule continues from the saved step).
3. **Save & Run All** again. The new version writes fresh `ckpt_latest.pt` / `ckpt_best.pt` / `log.csv` to its own output; `log.csv` is appended, not overwritten, when resuming.
4. Budget: a resume run should be capped at the hours remaining in the plan (`--time_limit_hours`), not 10 again.

---

*Next step: write `data/prepare_data.py` and Notebook 01, then `model.py` + `train.py` with the smoke config.*
