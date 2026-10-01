# PyGPT — slide deck outline (13 slides, 7–8 min talk + 2 min demo)

Numbers marked `⟨…⟩` come from the real run: fill from `results/metrics.json`,
`loss_curve.png`, and Notebook 02's bench output. Criterion mapping is PROJECT.md §12.

---

## 1 · Title
**PyGPT: a Python code-completion assistant, trained from scratch**
Rehan · Machine Learning · Bennett University · Oct 2026
One-liner at the bottom: *"Every stage — data → tokenizer → model → training → evaluation → deployment — built and measured by hand."*

## 2 · Problem & end goal
- Task: given the start of a Python function, predict the continuation (next-token prediction *is* code completion).
- Goal: a live web app that completes code in < 15 s, powered by a model **we** trained — not an API, not a fine-tune.
- "Working" = live demo + measured results vs baselines + repo that matches this deck.

## 3 · Existing systems & the gap (C1)
Table (from PROJECT.md §3.1): GPT-2 (2019, 124M, English) · Codex/Copilot (2021, 12B, closed) · CodeParrot (2021, 110M/1.5B, open, same dataset family) · StarCoder (15.5B) · Code Llama (70B) · **PyGPT (ours, 111M, ~0.5B tokens)**.
**Gap:** everything is either closed or needs 100–10,000× our compute. None teaches you how a code LM is *built*. PyGPT = complete, reproducible, honest pipeline at student scale.

## 4 · Feasibility (C1)
- Compute math: 6·N·D = 6 × 1.1e8 × 5e8 ≈ **3.3e17 FLOPs** → ~3.5–6 h on Kaggle's 2× T4 at 20–30% utilization. Fits a 10 h session.
- Measured: ⟨tok/s from bench⟩ tok/s → ⟨N⟩M tokens in ⟨H⟩ h. Chinchilla-optimal would be ~2.2B tokens: we are ~3–7× under-trained **by design** (limitations slide).
- War story: AWS GPU quota denied twice → Kaggle free tier (30 GPU-h/week) was sufficient. Budget table: data 0 h (CPU), smoke 2 h, main run 10 h, reserve 11 h.

## 5 · Objectives O1–O7 (C2)
The measurable table from PROJECT.md §4.1 — one row each, green check marks added live:
O1 model from scratch (init loss 10.4 ✓ causal mask ✓) · O2 corpus (⟨…⟩M tokens ✓) · O3 val loss < 1.8 (⟨…⟩) · O4 eval vs baselines (⟨…⟩) · O5 deployed app (⟨URL⟩) · O6 design↔code sync (slide 11) · O7 stretch: VS Code.

## 6 · Methodology pipeline (C2)
The §4.2 diagram, one box per stage, each labeled with its file:
HF stream → `prepare_data.py` (filter + BPE) → `train.bin`/`val.bin` → `train.py` (DDP 2×T4, fp16) → `ckpt_best.pt` → `eval.py` → `export.py` → `app/app.py` (Gradio). Call out: streaming avoids a 50 GB download; uint16 memmap because vocab 32,768 fits exactly.

## 7 · Architecture (C3, C4)
Block diagram of `model.py`: tokens → wte + wpe → 12 × Block(LN → CausalSelfAttention → LN → MLP) → LN → lm_head (tied).
Param breakdown: embeddings 25.2M + 12 blocks × 7.1M ≈ **111M** ("GPT-2 Small class; 124M with GPT-2's 50k vocab — ours is smaller because the code tokenizer has 32,768 tokens").
Why each piece (one line each): pre-LN = stable at depth · weight tying = −25M params · SDPA kernel = memory-efficient attention on T4 · ctx 512 = 1.4× faster, enough for functions.

## 8 · Data, tokenizer, training recipe (C3)
- Data: `codeparrot-clean` (dedup'd Python from GitHub), streamed ⟨…⟩M tokens, light filters (size, line length, % alpha, auto-generated).
- Tokenizer: byte-level BPE, 32k, code-trained → ~30–40% better compression of Python than GPT-2's English vocab (show one tokenized snippet).
- Recipe: AdamW (0.9, 0.95) wd 0.1 matrices-only · LR 6e-4 warmup 200 → cosine 6e-5 · fp16 + GradScaler · grad-accum → 131k tokens/step · clip 1.0 · DDP.
  Each row gets a one-word "why" (stability / throughput / memory / generalization).

## 9 · Results (C2, C3)
- `loss_curve.png`: 10.4 → ⟨final⟩ in ⟨…⟩ h / ⟨…⟩M tokens.
- Table: | metric | random init | GPT-2 124M | **PyGPT** |: val ppl ⟨32768 / — / …⟩ · AST-valid % ⟨~0 / … / …⟩ · pass@1 ⟨…⟩ · pass@10 ⟨…⟩.
- 2–3 short samples from `samples.md` (one good, one mediocre). Honesty line: "plausible > correct at this scale."

## 10 · Live demo (C4)
Switch to the Space (pre-warmed). Prompts A (fibonacci) → B (NumPy) → C (class Stack) → temperature 0.2 vs 0.9 → failure case F (sudoku).
Backup: laptop `python app/app.py` → video → screenshots (in this order).

## 11 · Design ↔ implementation (C4)
Left: repo tree. Right: §6.2 table — design component → file → symbol → how it was verified
(e.g. causal mask → `model.py:CausalSelfAttention` → sanity check 2: "changing token t+1 never changes logits ≤ t").
Line: "Every box on slide 7 names the class that implements it; the repo is public."

## 12 · Limitations & future work (C1)
Under-trained vs Chinchilla (~3–7×) · 512 context · no instruction tuning ("write a function that…" won't work) · completion ≠ correctness (pass@k shows it) · future: own tokenizer, longer training, FIM objective, VS Code extension ⟨or demo it if done⟩.

## 13 · References
Vaswani 17 · Radford 19 (GPT-2) · Chen 21 (Codex/HumanEval) · Kocetkov 22 / Li 23 (Stack/StarCoder) · Rozière 23 (Code Llama) · Kaplan 20 / Hoffmann 22 (scaling) · Sennrich 16 (BPE) · Loshchilov 19 (AdamW) · Micikevicius 18 (fp16) · Press & Wolf 17 (tying) · Holtzman 20 (nucleus) · Karpathy nanoGPT.

---

### Build notes
- Export PPTX **and** PDF; embed demo video + screenshots after Day 2.
- Prepared one-liners: why not fine-tune? (criterion 3 — show the machinery; trade-off acknowledged) · why 512 ctx? (speed/memory on T4; functions fit) · why 111M ≠ 124M? (32k code vocab vs 50k English vocab) · 10× compute? (Chinchilla: ~2.2B tokens first, then scale params).
