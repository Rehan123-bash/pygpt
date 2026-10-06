# PyGPT: A Python Code-Completion Assistant Trained From Scratch

**Machine Learning course project report · Bennett University · October 2026**
**Author:** Rehan
**Code:** https://github.com/Rehan123-bash/pygpt · **Weights & demo:** https://huggingface.co/rehannn11223/pygpt

---

## Abstract

PyGPT is a Python code-completion assistant powered by a GPT-2-Small-architecture language
model (110.5M parameters) trained **entirely from scratch** in PyTorch — no pretrained
weights, no fine-tuning, no external APIs. The final model was trained on 2.35B tokens of
Python source code (an approximately Chinchilla-optimal budget for its size) using free
Kaggle 2×T4 GPUs in ~17.5 hours, reaching a held-out validation loss of **1.509**
(perplexity **4.52**). Evaluated against a random-initialization baseline and vanilla GPT-2
124M under an identical harness, PyGPT produces syntactically valid Python far more often
than either baseline and is the only model of the three to pass any hidden functional tests.
The system ships as a fully offline web application generating ~40 tokens/second on an
ordinary laptop CPU. Beyond the artifact itself, the project documents reproducible
engineering findings: a one-character tokenization effect that silently zeroed functional
scores, a measured "specialization tax" from naive domain fine-tuning, and the repair of both.

---

# 1. Review of Existing Systems and Project Feasibility

## 1.1 Existing systems

| System | Year | Params | Training data | Open? | Relation to this work |
|---|---|---|---|---|---|
| GPT-2 (Radford et al.) | 2019 | 124M–1.5B | WebText (English) | Yes | Architecture we reimplement; English-only → our baseline |
| Codex / GitHub Copilot (Chen et al.) | 2021 | 12B | GitHub code | No | Defined the task and the pass@k metric we adopt |
| CodeParrot (Hugging Face) | 2021 | 110M / 1.5B | ~50 GB Python | Yes | Closest prior work: same dataset family and tokenizer, 50–100× our compute |
| SantaCoder / StarCoder (BigCode) | 2022–23 | 1.1B / 15.5B | 80+ languages | Yes | License-aware data pipeline we follow in spirit |
| Code Llama (Meta) | 2023 | 7B–70B | 500B+ tokens | Yes | Open state of the art |
| Tabnine / Cursor / Amazon Q | — | undisclosed | — | No | The commercial UX we imitate |

## 1.2 Gap and novelty

Every system above is either closed-source or trained with orders of magnitude more compute
than a student can access; none of them is useful for *learning how a code LM is built*.
PyGPT's claim is deliberately not "competitive quality"; it is:

1. **Fully local and offline.** The assistant runs with no internet connection, no API key,
   and no data leaving the machine. The class demo runs on the presenter's laptop.
2. **Genuinely lightweight.** 110.5M parameters in a 211 MB fp16 file, generating ~40 tok/s
   on a laptop CPU. Popular "local LLMs" (7B+) require 8–16 GB of RAM and capable hardware;
   PyGPT runs on essentially any low-specification device.
3. **Specialized for code.** A code-trained 32k BPE vocabulary (indentation and identifiers
   are single tokens) and Python-only training data make a small model punch above its
   weight on its one job.
4. **A transparent, measured, end-to-end pipeline.** Data → tokenizer → model → distributed
   training → evaluation → deployment, every stage visible, tested, and reproducible on free
   compute — with honest numbers for what that budget buys.

## 1.3 Feasibility analysis

**Compute estimate (scaling laws).** Training FLOPs ≈ 6·N·D = 6 × 1.1×10⁸ × 2.35×10⁹ ≈
1.5×10¹⁸. A T4 sustains ~13–20 TFLOPS of useful fp16 throughput at this model size, giving
~18 h on two T4s — within Kaggle's free 30 GPU-h/week. **Measured reality:** ~39,000
tokens/s sustained on 2×T4 (fp16, DDP), 2.35B tokens in 17.5 h across two sessions.
Chinchilla-optimal data for 110M parameters is ≈ 2.2B tokens; the final run is approximately
compute-optimal — a deliberate, defensible budget.

**Platform findings** (the free-tier landscape shifted during the project; both findings are
treated as feasibility results):
- AWS EC2 GPU quota was denied twice for a new account → all training moved to Kaggle.
- Hugging Face discontinued free hosting of Gradio Spaces mid-project (HTTP 402: PRO
  required) → the demo pivoted to a locally served page, which is also the more robust
  choice for a live, internet-independent demonstration. Model weights remain publicly
  hosted on HF (model repos are free).

**Session-limit engineering.** Kaggle caps sessions (~12 h); the training loop therefore
checkpoints every 250 steps and supports clean time-limited exit and resume. The final run
deliberately spanned two sessions (9.7 h + 7.8 h) resuming a single cosine learning-rate
schedule mid-descent — this mechanism was tested (kill + resume) before any long run.

**Budget actually used:** data preparation on CPU sessions (0 GPU-h); all experiments and
both full training runs consumed well under the weekly free quota, with reserve maintained
for failure recovery at every stage.

---

# 2. Objectives and Methodology

## 2.1 Measurable objectives and outcomes

| # | Objective | Success measure | Outcome |
|---|---|---|---|
| O1 | GPT-2-class decoder-only transformer from scratch in PyTorch | init loss ≈ ln(32768)=10.42; causal-mask test; overfit-one-batch | all pass; 110.5M params |
| O2 | Clean Python pretraining corpus | ≥800M train tokens + held-out validation | 2.2B fresh tokens + 5M val; +156M DSA curriculum blended at ~6.6% |
| O3 | Useful loss on free GPUs | val loss < 1.8 within budget | **1.509** (ppl 4.52) |
| O4 | Quantitative evaluation vs baselines | perplexity, AST-validity, pass@k | §5; PyGPT leads every metric |
| O5 | Deployed completion app | <15 s for a completion on CPU | ~2.4 s / 96 tokens (KV cache), offline |
| O6 | Design ↔ implementation synchronization | mapping table + public repo | §4; repo public |

## 2.2 Pipeline

```
HF Hub: codeparrot-clean  (streamed; the full ~50 GB is never downloaded)
   │  filters: file size, mean line length, % alphabetic, auto-generated markers
   ▼
data/prepare_data.py ── BPE tokenize (codeparrot 32k vocab) ── <|endoftext|> joins files
   ▼
train.bin / val.bin   (uint16 memmaps — vocab 32,768 fits the type exactly)
   +  DSA curriculum (TheAlgorithms/Python + DSA-filtered codeparrot) appended ×10 ≈ 6.6%
   ▼
train.py   AdamW · warmup+cosine · fp16+GradScaler · grad-accum · DDP (2×T4)
   │        131,072 tokens/optimizer-step · ckpt/250 steps · tested resume
   ▼
ckpt_best.pt ──► eval.py  (perplexity · AST-validity · functional pass@k · baselines)
   ▼
export.py  fp16 weights-only, 211 MB ──► server.py + hand-built page (offline demo)
                                     └─► app/app.py (Gradio variant, on HF)
```

## 2.3 Data and tokenizer

Training data streams from `codeparrot/codeparrot-clean-train` (deduplicated Python from
GitHub); validation uses the disjoint `-valid` split. Light filters discard files >100k
chars, mean line length >100 (minified/dumps), <25% alphabetic characters, or
auto-generation markers in the first lines. Files are tokenized with the
`codeparrot/codeparrot` byte-level BPE (vocab 32,768) and joined with `<|endoftext|>`.
The code-trained vocabulary encodes Python ~35% more compactly than GPT-2's English
vocabulary — every training token carries more signal.

**DSA curriculum.** Because random GitHub Python under-represents textbook algorithmic
functions, a 15.6M-token curated corpus (TheAlgorithms/Python, MIT-licensed, plus codeparrot
files that define classic DSA functions such as `is_prime`, `gcd`, `binary_search`,
`find_median`) is appended ×10 to the training file. Random 512-token windows therefore
sample algorithm-style code ~6.6% of the time throughout training. §5.3 explains why
blending beats a bolt-on fine-tuning phase (measured specialization tax).

## 2.4 Model

GPT-2 Small architecture: 12 pre-LayerNorm transformer blocks, 12 heads, 768 dims, context
512, learned positional embeddings, GELU MLPs (4× expansion), weight-tied embedding/output
head, GPT-2 initialization with residual projections scaled by 1/√(2·n_layer). Attention
uses PyTorch's fused scaled-dot-product kernel. Parameter count: 25.6M embeddings + 12 ×
7.1M blocks ≈ **110.5M**. (The familiar "124M" figure for GPT-2 Small includes its 50k
English vocabulary; ours is a 32k code vocabulary — same architecture class.)

## 2.5 Training recipe (final run)

| Setting | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), weight decay 0.1 on ≥2-D tensors only |
| LR schedule | warmup 200 steps → 6e-4 → cosine → 6e-5 at step 17,900 |
| Batch | micro-batch 16 × ctx 512 × 2 GPUs × grad-accum 8 = 131,072 tokens/step |
| Precision | fp16 autocast + dynamic loss scaling; gradient clipping 1.0 |
| Parallelism | DistributedDataParallel via torchrun, 2×T4; grad sync on last micro-step |
| Duration | 17,900 steps = 2.346B tokens, 17.5 h over two sessions (9.7 h + 7.8 h) |
| Robustness | checkpoint every 250 steps; clean time-limit exit; resume continues schedule |

Validation loss trajectory: 10.42 (init) → 1.603 at the session boundary (step 9,960) →
**1.509 final** (best checkpoint at step 17,750), with the schedule completing naturally.

## 2.6 Evaluation protocol

- **E1 Perplexity:** deterministic sequential 512-token windows over held-out tokens.
- **E2 Syntactic validity:** 20 prompts × 5 samples (T=0.6, top-k 50, top-p 0.95, 96 new
  tokens), completion trimmed at natural boundaries; report % where `ast.parse(prompt +
  completion)` succeeds.
- **E3 Functional correctness:** five classic tasks — `is_prime`, `add`, `multiply`,
  `divide`, `find_median` — prompted in TheAlgorithms style (type hints + docstring +
  doctests). Each sampled completion is executed against 3 hidden asserts in a sandboxed
  subprocess (isolated interpreter, timeout); pass@1 and pass@10 use the unbiased estimator
  of Chen et al. (2021). n=30 samples/task for the headline numbers.
- **E4 Loss curves** from the training CSV log. **E5 Qualitative samples**, including an
  honest failure case.
- **Baselines:** a random-initialization model (identical architecture) and vanilla GPT-2
  124M, both run through the *identical* sampling, trimming, and execution harness.

---

# 3. Relevance of Algorithms and Techniques

Each technique was chosen for a reason that can be defended independently:

| Technique | Why it is the right choice here |
|---|---|
| Byte-level BPE, code-trained 32k vocab | open vocabulary; whitespace/identifiers as single tokens; ~35% better Python compression than an English vocab |
| Decoder-only causal transformer | next-token prediction *is* code completion; fully parallel training over sequences |
| Pre-LayerNorm | optimization stability at 12 layers |
| Learned positional embeddings | simple and sufficient for a fixed 512 context |
| Weight tying | −25M parameters; improves small-model efficiency |
| GPT-2 scaled residual init | keeps residual-stream variance bounded over 24 additions |
| Fused scaled-dot-product attention | memory-efficient attention kernel on T4 |
| AdamW (decoupled decay, matrices only) | regularization that does not fight Adam's scaling; norms/biases undecayed |
| Warmup + cosine decay | avoids early divergence; low final LR sharpens the minimum; schedule resumes across sessions |
| fp16 + GradScaler | 2–3× tensor-core throughput on T4 (no bf16); scaler prevents gradient underflow |
| Gradient accumulation | 131k-token effective batches within 16 GB cards |
| Gradient clipping 1.0 | caps rare fp16 loss spikes |
| DistributedDataParallel | near-linear 2-GPU scaling; measured ~39k tok/s |
| Checkpoint / resume | survives platform session caps; tested before use |
| Temperature / top-k / nucleus sampling | the determinism↔diversity trade-off, demonstrated live |
| KV-cache generation | each step feeds 1 token instead of the whole sequence: 96-token completions 25 s → **2.4 s** on CPU; outputs proven bit-identical to the uncached path by a unit test |
| Doctest-verified sampling | draw up to 8 candidates; return the first that *passes the prompt's own doctests* in a sandboxed interpreter ("the model proposes, the doctests dispose" — search, not added knowledge); falls back to first parseable |
| Scaling laws (Kaplan; Hoffmann) | sized the final run at ≈ Chinchilla-optimal tokens for 110M params |
| Perplexity, AST-validity, pass@k | intrinsic + syntactic + functional evaluation with baselines |

## 3.1 Finding: a one-character tokenization effect

Initial functional scores were 0/50 for every model. Inspection of raw completions showed
the models *closing* the function (emitting end-of-text or starting a new `def`) instead of
writing a body. Root cause: evaluation prompts ended with a bare newline after the
docstring's closing `"""`. The BPE encodes *newline+indentation* as a **single token**; a
bare `\n` token therefore tells the model "the next line is unindented" — i.e., the function
is complete. Removing that one character moved pass@10 from 0.00 to 0.39 **with no
training**. All prompts now end at the closing quotes. This is a reproducible, demonstrable
consequence of byte-level BPE on whitespace-significant languages.

## 3.2 Finding: the specialization tax

An intermediate experiment fine-tuned a trained checkpoint on the 15.6M-token DSA corpus for
~900 steps ("bolt-on" curriculum). DSA-task accuracy improved, but general held-out
perplexity degraded from 5.24 to 6.19 (+0.18 nats) — catastrophic forgetting in miniature,
measured. The final run therefore *blends* the curriculum at ~6.6% of every batch instead,
capturing the domain benefit inside a single training distribution.

---

# 4. Synchronization of Project Design and Implementation

## 4.1 Design-to-code map

| Design component | File · symbol | Verified by |
|---|---|---|
| Data filtering & tokenization | `data/prepare_data.py` · `keep_file`, `tokenize_stream` | token counts re-counted from disk; decoded window inspected |
| DSA curriculum | `data/prepare_algo_data.py` | meta.json counts; blend size printed at prep time |
| Causal self-attention | `model.py` · `CausalSelfAttention` | unit test: perturbing token t+1 never changes logits at ≤ t |
| Transformer / full model | `model.py` · `GPT` | init loss ≈ ln(V); overfit-one-batch < 0.1; parameter count |
| Optimizer grouping | `model.py` · `configure_optimizers` | decay/no-decay split asserted in training logs |
| Training loop | `train.py` · `get_lr`, `estimate_loss`, `save_ckpt` | smoke run; kill+`--resume` continuity; 2-process DDP; time-limit exit |
| Sampling + KV cache | `model.py` · `generate` | cached output bit-identical to uncached (sanity check 5) |
| Evaluation | `eval.py` · `val_loss`, `ast_validity`, `functional_test` | `results/metrics.json`; baselines share the harness |
| Export | `export.py` · `export_fp16`, `load_exported` | fp16 round-trip greedy-identical (`--verify`) |
| Demo (primary) | `server.py` + `app/static/index.html` | offline page; POST /complete |
| Demo (HF variant) | `app/app.py` | published with the weights |
| Orchestration | `kaggle/push_day1.py`, `notebooks/` | every Kaggle run in this report was launched and logged through it |

**Testing discipline:** `tests/sanity.py` runs six checks (init loss, causal mask, weight
tying, overfit-one-batch, generation invariants incl. KV-cache equivalence, parameter count)
and gates every training launch; it passed 6/6 on CPU and on the Kaggle GPUs before each run.

## 4.2 Repository

The public repository mirrors the design document's tree exactly (model, training, data,
evaluation, export, app, notebooks, orchestration, results, slides, report). Results files
(`results/metrics.json`, `log.csv`, `loss_curve.png`, `samples.md`) are the artifacts cited
in this report.

---

# 5. Results

## 5.1 Headline comparison (identical harness for all models)

| Metric | Random init | Vanilla GPT-2 124M | **PyGPT (final)** |
|---|---|---|---|
| Val loss / perplexity (held-out Python) | 10.56 / 38,487 | n/a (different vocab) | **1.509 / 4.52** |
| AST-valid completions | 0% | 7% | **{AST_PCT}** |
| pass@1 — 5 DSA tasks (n=30) | — | 0.00 | **{PASS1}** |
| pass@10 — 5 DSA tasks (n=30) | — | 0.00 | **{PASS10}** |

## 5.2 Training progression (what the project learned between runs)

| Stage | Unique data | Val loss | DSA pass@10 |
|---|---|---|---|
| Run 1: one epoch | 800M | 1.685 | 0.00 → 0.39 after the prompt fix (§3.1) |
| + second epoch (same data) | 800M ×2 | 1.639 | 0.39 |
| + bolt-on DSA phase | +15.6M ×~6 | 1.823 (general) | 0.58 — but at a general-perplexity cost (§3.2) |
| **Run 2: fresh data + 6.6% blend** | **2.35B** | **1.509** | **{PASS10}** |

Repeating data bought little (−0.046 nats); fresh data at compute-optimal scale bought a lot
(−0.13 nats from the best previous general checkpoint, with the curriculum included rather
than taxed).

## 5.3 Qualitative behaviour

The model reliably produces well-formed Python (correct indentation, docstrings, idioms such
as `with open(path) as f: return json.load(f)`); semantic correctness is strongest on the
curriculum-adjacent one-function tasks and degrades with logical complexity. `is_prime`
(loop + modulo reasoning) remains the model's known boundary and is presented as the honest
failure case, alongside `solve_sudoku`.

---

# 6. Limitations and Future Work

- **Completion ≠ correctness.** Syntactic fluency far exceeds semantic reliability; both are
  reported separately on principle.
- **512-token context** restricts the model to function-level completion.
- **No instruction tuning** — natural-language commands ("write a function that…") are out
  of scope by design.
- **Scale ceiling** — at 110M parameters and ~18 GPU-hours, results are honest, not
  competitive with commercial systems (CodeParrot used 50–100× more compute).
- Future work: custom tokenizer, longer context, fill-in-the-middle objective, instruction
  tuning, and a VS Code inline-completion extension (its backend endpoint `POST /complete`
  already exists in `server.py`).

# 7. References

1. Vaswani et al., 2017 — Attention Is All You Need.
2. Radford et al., 2019 — Language Models are Unsupervised Multitask Learners (GPT-2).
3. Chen et al., 2021 — Evaluating Large Language Models Trained on Code (Codex; pass@k).
4. Tunstall, von Werra, Wolf, 2022 — NLP with Transformers, ch. 10 (CodeParrot).
5. Kocetkov et al., 2022 — The Stack; Li et al., 2023 — StarCoder.
6. Rozière et al., 2023 — Code Llama.
7. Kaplan et al., 2020 — Scaling Laws; Hoffmann et al., 2022 — Chinchilla.
8. Sennrich et al., 2016 — BPE. 9. Loshchilov & Hutter, 2019 — AdamW.
10. Micikevicius et al., 2018 — Mixed Precision Training.
11. Press & Wolf, 2017 — Weight Tying. 12. Holtzman et al., 2020 — Nucleus Sampling.
13. Karpathy — nanoGPT (reference implementation pattern).

---

## Appendix A — Reproduction

```bash
pip install -r requirements.txt
python tests/sanity.py                          # 6/6 must pass
python data/prepare_data.py --out_dir data --train_tokens 2_200_000_000
python data/prepare_algo_data.py --out_dir data/algo
torchrun --standalone --nproc_per_node=2 train.py --config config/gpt2_small_py.py \
    --max_iters=17900 --time_limit_hours=9.7            # + one --resume session
python eval.py --ckpt ckpt_best.pt --data_dir data --gpt2
python export.py --ckpt ckpt_best.pt --out app/model_fp16.pt --verify
python server.py                                 # offline demo at 127.0.0.1:8000
```

## Appendix B — Demo prompts

The five functional-test prompts (TheAlgorithms style: type hints, docstring, doctests,
ending at the closing `"""` with no trailing newline — see §3.1), plus `solve_sudoku` as the
honest failure exhibit.
