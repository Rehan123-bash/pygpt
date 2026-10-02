"""
build_slides.py - Generate slides/pygpt_slides.pptx (13 slides, PROJECT.md section 12).
All numbers are from the real Oct 2 run (results/metrics.json). Regenerate after edits:
    python slides/build_slides.py
"""

import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DARK = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT = RGBColor(0x0F, 0x4C, 0x81)
GRAY = RGBColor(0x55, 0x55, 0x55)

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide_base(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(32), True, DARK
    if subtitle:
        p2 = tb.text_frame.add_paragraph()
        p2.text = subtitle
        p2.font.size, p2.font.color.rgb = Pt(14), GRAY
    return s


def bullets(s, items, left=0.6, top=1.5, width=12.1, height=5.6, size=18):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        level, text = item if isinstance(item, tuple) else (0, item)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("•  " if level == 0 else "–  ") + text
        p.level = level
        p.font.size = Pt(size - 2 * level)
        p.font.color.rgb = DARK if level == 0 else GRAY
        p.space_after = Pt(10)
    return tb


def mono(s, text, left=0.6, top=1.5, width=12.1, height=5.0, size=14):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = False
    for i, line in enumerate(text.strip("\n").split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name, p.font.size, p.font.color.rgb = "Consolas", Pt(size), DARK
    return tb


def table(s, headers, rows, left=0.5, top=1.5, width=12.3, height=None, size=13):
    h = height or min(5.6, 0.4 * (len(rows) + 1))
    shape = s.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top),
                               Inches(width), Inches(h))
    t = shape.table
    for j, head in enumerate(headers):
        c = t.cell(0, j)
        c.text = head
        c.text_frame.paragraphs[0].font.size = Pt(size)
        c.text_frame.paragraphs[0].font.bold = True
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i + 1, j)
            c.text = str(val)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(size)
    return t


# 1 ---------------------------------------------------------------- title
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(1), Inches(2.3), Inches(11.3), Inches(3))
p = tb.text_frame.paragraphs[0]
p.text = "PyGPT"
p.font.size, p.font.bold, p.font.color.rgb, p.alignment = Pt(66), True, ACCENT, PP_ALIGN.CENTER
for txt, sz, col in [
    ("A Python code-completion assistant, trained from scratch", 28, DARK),
    ("Every stage built and measured by hand: data → tokenizer → model → training → evaluation → deployment", 16, GRAY),
    ("", 14, GRAY),
    ("Rehan  ·  Machine Learning  ·  Bennett University  ·  October 2026", 18, DARK),
]:
    q = tb.text_frame.add_paragraph()
    q.text, q.alignment = txt, PP_ALIGN.CENTER
    q.font.size, q.font.color.rgb = Pt(sz), col

# 2 ---------------------------------------------------------------- problem
s = slide_base("The problem and the end goal")
bullets(s, [
    "Task: given the start of a Python function, predict what comes next — next-token prediction is code completion",
    "Goal: a working assistant powered by a model we trained ourselves — not an API call, not a fine-tune",
    "“Working” means three things:",
    (1, "a live demo that completes code in seconds"),
    (1, "measured results against baselines (perplexity, syntactic validity, functional tests)"),
    (1, "a public repo whose structure matches the design in this deck"),
    "Constraints: 3 days, free compute only (Kaggle 2× T4, 30 GPU-h/week), solo",
])

# 3 ---------------------------------------------------------------- existing systems
s = slide_base("Existing systems — and the gap", "Criterion 1")
table(s, ["System", "Year", "Params", "Open?", "Relevance"], [
    ["GPT-2 (Radford et al.)", "2019", "124M–1.5B", "Yes", "Architecture we reuse; English-only → our baseline"],
    ["Codex / Copilot (Chen et al.)", "2021", "12B", "No", "Defined the task and pass@k evaluation"],
    ["CodeParrot (HF)", "2021", "110M / 1.5B", "Yes", "Closest prior work: same data family, 50–100× our compute"],
    ["SantaCoder / StarCoder (BigCode)", "2022–23", "1.1B / 15.5B", "Yes", "License-filtered data pipeline we follow in spirit"],
    ["Code Llama (Meta)", "2023", "7B–70B", "Yes", "Open state of the art"],
    ["PyGPT (ours)", "2026", "110.5M", "Yes", "Complete, reproducible, measured pipeline at student scale"],
], size=12)
tb = s.shapes.add_textbox(Inches(0.6), Inches(5.6), Inches(12), Inches(1.4))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = ("Gap: everything above is closed-source or needs 100–10,000× a student's compute. "
          "None of it teaches how a code LM is built. PyGPT's contribution is honest numbers on "
          "what ~7 GPU-hours buys — with every stage visible.")
p.font.size, p.font.italic, p.font.color.rgb = Pt(16), True, ACCENT

# 4 ---------------------------------------------------------------- feasibility
s = slide_base("Feasibility: do the math before burning GPU hours", "Criterion 1")
bullets(s, [
    "Scaling-law estimate: FLOPs ≈ 6·N·D = 6 × 1.1e8 × 8e8 ≈ 5.3e17 → ~6 h on 2× T4 at 20–30% utilization",
    "Measured on the day: 34,000 tok/s (fp16, DDP) → 799.5M tokens in 6 h 45 min — one full epoch",
    "Chinchilla-optimal for 110M params ≈ 2.2B tokens → we are ~3× under-trained by design (limitations slide)",
    "The free-tier landscape is a moving target — verify, don't assume:",
    (1, "AWS: GPU quota denied twice (new account) → dead end"),
    (1, "Hugging Face: free Gradio Spaces discontinued mid-project (402) → laptop demo instead"),
    (1, "Kaggle: 30 GPU-h/week free — total project spend 7.3 h, leaving 22.7 h reserve"),
    "Session-death insurance: checkpoint every 250 steps + tested kill-and-resume before the long run",
])

# 5 ---------------------------------------------------------------- objectives
s = slide_base("Objectives — all measurable, all measured", "Criterion 2")
table(s, ["#", "Objective", "Target", "Achieved"], [
    ["O1", "GPT-2-class model from scratch in PyTorch", "init loss ≈ ln(32768)=10.4; causal-mask test", "10.42 ✓ · mask test ✓ · 110.5M params ✓"],
    ["O2", "Clean Python pretraining corpus", "≥800M train tokens + held-out val", "800M + 5M (codeparrot-clean) ✓"],
    ["O3", "Useful loss on free GPUs", "val loss < 1.8 in ≤ 10 GPU-h", "1.685 in 6.75 h ✓ (ppl 5.24)"],
    ["O4", "Quantitative eval vs baselines", "ppl, AST-validity, pass@k", "59% AST-valid vs GPT-2 10% ✓ · pass@10 = 0 (honest)"],
    ["O5", "Deployed completion app", "30-token prompt answered < 15 s", "2.4 s for 96 tokens on laptop CPU ✓"],
    ["O6", "Design ↔ implementation sync", "map table + public repo", "slide 11 + GitHub + hf.co/rehannn11223/pygpt ✓"],
    ["O7", "(stretch) VS Code extension", "60-s video", "future work"],
], size=12)

# 6 ---------------------------------------------------------------- pipeline
s = slide_base("Methodology: the pipeline", "Criterion 2 — every box is a file in the repo")
mono(s, """
 HF Hub: codeparrot-clean (streamed, never fully downloaded)
            │  filters: size, line length, %alpha, auto-generated
            ▼
 data/prepare_data.py ── BPE tokenize (32k code vocab) ── <|endoftext|> joins files
            ▼
 train.bin / val.bin  (uint16 memmap — vocab 32,768 fits exactly)
            ▼
 train.py   AdamW · warmup+cosine · fp16+GradScaler · grad-accum · DDP on 2× T4
            │        checkpoints every 250 steps · resume · CSV log
            ▼
 ckpt_best.pt ──► eval.py   val ppl · AST-validity · pass@k · baselines
            ▼
 export.py  fp16 weights-only (212 MB) ──► app/app.py  (Gradio, runs on a laptop CPU)
""", size=14)

# 7 ---------------------------------------------------------------- architecture
s = slide_base("Architecture: GPT-2 Small, written from scratch", "Criteria 3+4 — file and class names on every box")
mono(s, """
 tokens ──► wte (32768×768) + wpe (512×768)          model.py: GPT
              ▼
   12 × Block:   x = x + CausalSelfAttention(LN(x))  model.py: Block, CausalSelfAttention
                 x = x + MLP(LN(x))                  model.py: MLP   (768 → 3072 → 768, GELU)
              ▼
   LayerNorm ──► lm_head (tied to wte)               weight tying: −25M params
""", top=1.4, height=2.6, size=14)
bullets(s, [
    "Parameters: embeddings 25.6M + 12 blocks × 7.1M ≈ 110.5M  (“124M-class”: GPT-2's 50k English vocab vs our 32k code vocab)",
    "Why each choice: pre-LN → stable at 12 layers · SDPA kernel → memory-efficient attention on T4 · ctx 512 → 1.4× faster, functions fit",
    "Generation: temperature / top-k / nucleus sampling + per-layer KV cache (each step feeds 1 token, not the whole sequence)",
], top=4.3, size=15)

# 8 ---------------------------------------------------------------- data + recipe
s = slide_base("Data, tokenizer, training recipe — each row has a why", "Criterion 3")
table(s, ["Choice", "Value", "Why"], [
    ["Data", "codeparrot-clean, 800M tokens streamed", "pre-deduplicated Python; streaming avoids a 50 GB download"],
    ["Tokenizer", "codeparrot BPE, vocab 32,768", "code-trained: indentation + identifiers are single tokens (~35% fewer than GPT-2's)"],
    ["Optimizer", "AdamW β=(0.9, 0.95), wd 0.1 on matrices only", "decoupled decay regularizes without fighting Adam"],
    ["LR schedule", "warmup 200 → 6e-4 → cosine → 6e-5", "warmup avoids early divergence; low final LR sharpens the minimum"],
    ["Precision", "fp16 autocast + GradScaler", "2–3× throughput on T4 tensor cores; scaler prevents underflow"],
    ["Batching", "16 × 512 × 2 GPUs × grad-accum 8 = 131k tok/step", "large effective batch within 16 GB"],
    ["Multi-GPU", "DistributedDataParallel (torchrun)", "near-linear scaling on the 2 free T4s"],
    ["Safety", "grad clip 1.0 · ckpt/250 steps · tested resume", "fp16 spikes; Kaggle 12 h session cap"],
], size=12)

# 9 ---------------------------------------------------------------- results
s = slide_base("Results — against real baselines", "Criteria 2+3 · all numbers in results/metrics.json")
img = os.path.join(REPO, "results", "loss_curve.png")
if os.path.exists(img):
    s.shapes.add_picture(img, Inches(0.4), Inches(1.5), width=Inches(6.4))
table(s, ["Metric", "Random init", "GPT-2 124M", "PyGPT (ours)"], [
    ["Val perplexity", "38,487", "n/a (different vocab)", "5.24"],
    ["AST-valid samples", "0%", "10%", "59%"],
    ["pass@1 (5 tasks)", "—", "0", "0"],
    ["pass@10 (5 tasks)", "—", "0", "0"],
], left=7.0, top=1.6, width=6.0, size=13)
tb = s.shapes.add_textbox(Inches(7.0), Inches(4.1), Inches(6.0), Inches(3.0))
tf = tb.text_frame; tf.word_wrap = True
for i, (txt, sz) in enumerate([
    ("Loss 10.4 → 1.685 over one epoch (6.75 h).", 14),
    ("59% of completions parse as Python — 6× vanilla GPT-2. But 0/5 toy tasks pass hidden tests: "
     "7 GPU-hours buys plausible, not correct. CodeParrot used 50–100× more compute.", 14),
    ("Best sample: read_json → idiomatic `with open(path) as f: return json.load(f)` — "
     "one missing import from passing.", 13),
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = txt
    p.font.size = Pt(sz)
    p.font.color.rgb = DARK if i == 0 else GRAY
    p.space_after = Pt(8)

# 10 --------------------------------------------------------------- demo
s = slide_base("Live demo", "Criterion 4")
bullets(s, [
    "python app/app.py — runs offline on this laptop, ~2.4 s per completion (KV cache)",
    "Prompt A: def fibonacci(n) + docstring — read what it does, not what we hope it does",
    "Prompt B: NumPy normalize() — has it learned library idioms?",
    "Prompt C: class Stack with __init__ — multi-method structure",
    "Same prompt at temperature 0.2 vs 0.9 — sampling in one sentence",
    "Prompt F: solve_sudoku() — the honest failure case",
    (1, "backups, in order: Kaggle public share link · recorded video · screenshots"),
])

# 11 --------------------------------------------------------------- design-implementation
s = slide_base("Design ↔ implementation: nothing on these slides is vaporware", "Criterion 4")
table(s, ["Design component", "File · symbol", "Verified by"], [
    ["Data filtering + tokenization", "data/prepare_data.py · keep_file, tokenize_stream", "token counts re-counted from disk; decoded sample inspected"],
    ["Causal self-attention", "model.py · CausalSelfAttention", "test: changing token t+1 never changes logits ≤ t"],
    ["Full model + init", "model.py · GPT", "init loss 10.42 ≈ ln(V); overfits one batch to < 0.1; 110.5M params"],
    ["Training loop", "train.py · get_lr, estimate_loss, save_ckpt", "smoke run · kill + --resume continues · DDP on 2 GPUs · time-limit exit"],
    ["Sampling + KV cache", "model.py · generate", "cached output bit-identical to uncached (sanity check 5)"],
    ["Evaluation", "eval.py · val_loss, ast_validity, functional_test", "results/metrics.json — the numbers on slide 9"],
    ["Export + demo", "export.py, app/app.py", "fp16 round-trip greedy-identical; this demo"],
], size=12)
tb = s.shapes.add_textbox(Inches(0.6), Inches(6.3), Inches(12), Inches(0.8))
p = tb.text_frame.paragraphs[0]
p.text = "Code: github.com/Rehan123-bash/pygpt  ·  Weights + runnable demo: huggingface.co/rehannn11223/pygpt"
p.font.size, p.font.color.rgb = Pt(15), ACCENT

# 12 --------------------------------------------------------------- limitations
s = slide_base("Limitations and future work", "Criterion 1 — knowing the boundary is part of the result")
bullets(s, [
    "Under-trained ~3× vs Chinchilla-optimal (0.8B of ~2.2B tokens) — loss was still falling when we stopped",
    "Completion ≠ correctness: pass@10 = 0 on toy tasks; it autocompletes plausibly, it does not reason",
    "512-token context — function-level, not file-level, completion",
    "No instruction tuning: “write a function that…” prompts will not work",
    "Future work:",
    (1, "longer training — 22 unused free GPU-hours per week were left on the table"),
    (1, "train our own tokenizer · fill-in-the-middle objective · VS Code inline-completion extension (backend already exists)"),
])

# 13 --------------------------------------------------------------- references
s = slide_base("References")
refs = [
    "Vaswani et al. 2017 — Attention Is All You Need",
    "Radford et al. 2019 — Language Models are Unsupervised Multitask Learners (GPT-2)",
    "Chen et al. 2021 — Evaluating LLMs Trained on Code (Codex, HumanEval, pass@k)",
    "Tunstall, von Werra, Wolf 2022 — NLP with Transformers, ch. 10 (CodeParrot)",
    "Kocetkov et al. 2022 — The Stack · Li et al. 2023 — StarCoder",
    "Rozière et al. 2023 — Code Llama",
    "Kaplan et al. 2020 — Scaling Laws · Hoffmann et al. 2022 — Chinchilla",
    "Sennrich et al. 2016 — BPE · Press & Wolf 2017 — Weight Tying",
    "Loshchilov & Hutter 2019 — AdamW · Micikevicius et al. 2018 — Mixed Precision",
    "Holtzman et al. 2020 — The Curious Case of Neural Text Degeneration (nucleus sampling)",
    "Karpathy — nanoGPT (reference implementation pattern)",
]
bullets(s, refs, size=15)

out = os.path.join(HERE, "pygpt_slides.pptx")
prs.save(out)
print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
