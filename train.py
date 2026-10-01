"""
train.py - Training loop for the PyGPT language model.

Features (each maps to a row in PROJECT.md section 4.6):
  * memmap data loader with random 512-token windows
  * AdamW with decoupled weight decay (matrices only)
  * linear warmup + cosine LR decay
  * fp16 autocast + dynamic loss scaling (GradScaler); falls back to fp32 on CPU
  * gradient accumulation, gradient clipping
  * DistributedDataParallel via torchrun (single process also works)
  * periodic eval, CSV logging, checkpoints (latest + best), --resume
  * --time_limit_hours: saves and exits cleanly before Kaggle kills the session

Usage:
  python train.py --config config/smoke.py
  torchrun --standalone --nproc_per_node=2 train.py --config config/gpt2_small_py.py
  python train.py --config config/gpt2_small_py.py --resume=/kaggle/input/.../ckpt_latest.pt
Any config value can be overridden on the CLI as --key=value.
"""

import ast
import csv
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import all_reduce, destroy_process_group, init_process_group, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# 1. Configuration: defaults <- config file <- CLI overrides
# -----------------------------------------------------------------------------
cfg = dict(
    # model
    n_layer=12, n_head=12, n_embd=768, block_size=512, vocab_size=32768, dropout=0.0, bias=False,
    # data
    data_dir="data", micro_batch_size=16, grad_accum_steps=8,
    # optimizer / schedule
    learning_rate=6e-4, min_lr=6e-5, warmup_iters=200, max_iters=3800,
    beta1=0.9, beta2=0.95, weight_decay=0.1, grad_clip=1.0,
    # system
    dtype="float16", compile=False, eval_interval=250, eval_iters=40, log_interval=10,
    ckpt_dir="runs/default", time_limit_hours=10.0, seed=1337,
    resume="",            # path to ckpt_latest.pt to continue from
    config="",            # path to a config .py file
)


def parse_cli(argv):
    """--config PATH / --config=PATH, then --key=value overrides (values parsed as Python literals)."""
    i = 0
    overrides = {}
    while i < len(argv):
        a = argv[i]
        if a in ("--config",) and i + 1 < len(argv):
            overrides["config"] = argv[i + 1]; i += 2; continue
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
        elif a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            k, v = a[2:], argv[i + 1]; i += 1
        else:
            raise SystemExit(f"unrecognised argument: {a}")
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass  # keep as string
        overrides[k] = v
        i += 1
    return overrides


cli = parse_cli(sys.argv[1:])
if cli.get("config"):
    ns = {}
    with open(cli["config"]) as f:
        exec(f.read(), ns)
    for k, v in ns.items():
        if k in cfg and not k.startswith("__"):
            cfg[k] = v
for k, v in cli.items():
    if k not in cfg:
        raise SystemExit(f"unknown config key: {k}")
    cfg[k] = v
globals().update(cfg)   # expose as plain names below

# -----------------------------------------------------------------------------
# 2. Distributed / device setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get("RANK", -1)) != -1          # set by torchrun
if ddp:
    init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(device)
    master = ddp_rank == 0
    seed_offset = ddp_rank
else:
    ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    master = True
    seed_offset = 0
device_type = "cuda" if device.startswith("cuda") else "cpu"

tokens_per_step = micro_batch_size * block_size * grad_accum_steps * ddp_world_size

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# fp16 only makes sense on CUDA; CPU trains in fp32
if device_type == "cpu" and dtype in ("float16", "bfloat16"):
    dtype = "float32"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
autocast_ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=ptdtype)
# GradScaler is only needed for fp16 (bf16/fp32 have enough exponent range)
_scaler_enabled = (dtype == "float16" and device_type == "cuda")
try:
    scaler = torch.amp.GradScaler("cuda", enabled=_scaler_enabled)        # torch >= 2.3
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=_scaler_enabled)           # older torch

if master:
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"device={device} world_size={ddp_world_size} dtype={dtype} tokens/step={tokens_per_step:,}")

# -----------------------------------------------------------------------------
# 3. Data: uint16 token streams, random windows
# -----------------------------------------------------------------------------
train_path = os.path.join(data_dir, "train.bin")
val_path = os.path.join(data_dir, "val.bin")
for p in (train_path, val_path):
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} - run data/prepare_data.py (or tests/make_synthetic_data.py) first")

meta_path = os.path.join(data_dir, "meta.json")
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("vocab_size") and meta["vocab_size"] != vocab_size and master:
        print(f"WARNING: meta.json vocab_size={meta['vocab_size']} != config vocab_size={vocab_size}")


def get_batch(split):
    # re-open the memmap each call to avoid a memory leak with np.memmap + many epochs
    data = np.memmap(train_path if split == "train" else val_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (micro_batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# 4. Model, optimizer, (optional) resume
# -----------------------------------------------------------------------------
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  vocab_size=vocab_size, dropout=dropout, bias=bias)
iter_num = 0
tokens_seen = 0
best_val_loss = float("inf")
checkpoint = None

if resume:
    if master:
        print(f"resuming from {resume}")
    checkpoint = torch.load(resume, map_location=device)
    # architecture must match the checkpoint, whatever the config says
    for k in ("n_layer", "n_head", "n_embd", "block_size", "vocab_size", "bias"):
        model_args[k] = checkpoint["model_args"][k]

model = GPT(GPTConfig(**model_args)).to(device)
if master:
    print(f"model: {model.num_params()/1e6:.1f}M params")

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if checkpoint is not None:
    state = checkpoint["model"]
    # strip torch.compile's prefix if the checkpoint was saved from a compiled model
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])
    iter_num = checkpoint["iter_num"]
    tokens_seen = checkpoint.get("tokens_seen", iter_num * tokens_per_step)
    best_val_loss = checkpoint["best_val_loss"]
    del checkpoint, state

if compile:
    if master:
        print("compiling model (first step will be slow)...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank] if device_type == "cuda" else None)
raw_model = model.module if ddp else model


# -----------------------------------------------------------------------------
# 5. Helpers: LR schedule, evaluation, checkpointing, logging
# -----------------------------------------------------------------------------
def get_lr(it):
    """Linear warmup to learning_rate, then cosine decay to min_lr at max_iters."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss():
    """Mean loss over eval_iters random batches for train and val; averaged across DDP ranks."""
    out = {}
    raw_model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with autocast_ctx:
                _, loss = raw_model(X, Y)
            losses[k] = loss.item()
        mean = losses.mean()
        if ddp:
            all_reduce(mean, op=ReduceOp.AVG)
        out[split] = mean.item()
    raw_model.train()
    return out


def save_ckpt(path, val_loss):
    """Atomic save: write to a temp file then rename, so a killed session never leaves a corrupt ckpt."""
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "model_args": model_args,
        "iter_num": iter_num,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "config": cfg,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


log_path = os.path.join(ckpt_dir, "log.csv")
if master:
    new_log = not (resume and os.path.exists(log_path))
    log_file = open(log_path, "a" if not new_log else "w", newline="")
    log_writer = csv.writer(log_file)
    if new_log:
        log_writer.writerow(["step", "tokens", "train_loss", "val_loss", "lr", "tok_per_sec", "elapsed_s"])

# -----------------------------------------------------------------------------
# 6. Training loop
# -----------------------------------------------------------------------------
t_start = time.time()
t_last = t_start
time_limit_s = time_limit_hours * 3600
X, Y = get_batch("train")      # prefetch first batch
running_loss = None
model.train()


def should_stop_for_time():
    """All ranks must agree, so rank 0 decides and broadcasts."""
    flag = torch.tensor([1.0 if (time.time() - t_start) > time_limit_s else 0.0], device=device)
    if ddp:
        all_reduce(flag, op=ReduceOp.MAX)
    return flag.item() > 0


try:
    while True:
        lr = get_lr(iter_num)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- evaluate + checkpoint ----
        if iter_num % eval_interval == 0 and (iter_num > 0 or not resume):
            losses = estimate_loss()
            if master:
                elapsed = time.time() - t_start
                print(f"[eval] step {iter_num} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                      f"lr {lr:.2e} | tokens {tokens_seen/1e6:.1f}M | {elapsed/60:.1f} min")
                if device_type == "cuda":
                    print(f"       peak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
                log_writer.writerow([iter_num, tokens_seen, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                                     f"{lr:.3e}", "", f"{elapsed:.0f}"])
                log_file.flush()
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        save_ckpt(os.path.join(ckpt_dir, "ckpt_best.pt"), losses["val"])
                        print(f"       saved ckpt_best.pt (val {best_val_loss:.4f})")
                if iter_num > 0:
                    save_ckpt(os.path.join(ckpt_dir, "ckpt_latest.pt"), losses["val"])

        if iter_num >= max_iters:
            break

        # ---- forward/backward with gradient accumulation ----
        loss_accum = 0.0
        for micro in range(grad_accum_steps):
            if ddp:
                # only sync gradients on the last micro-step
                model.require_backward_grad_sync = (micro == grad_accum_steps - 1)
            with autocast_ctx:
                _, loss = model(X, Y)
                loss = loss / grad_accum_steps
            X, Y = get_batch("train")          # prefetch next batch while GPU works
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1
        tokens_seen += tokens_per_step
        running_loss = loss_accum if running_loss is None else 0.9 * running_loss + 0.1 * loss_accum

        # ---- logging ----
        if iter_num % log_interval == 0 and master:
            now = time.time()
            dt = now - t_last
            t_last = now
            tok_s = tokens_per_step * log_interval / dt
            print(f"step {iter_num} | loss {loss_accum:.4f} (ema {running_loss:.4f}) | lr {lr:.2e} | "
                  f"{dt/log_interval*1000:.0f} ms/step | {tok_s:,.0f} tok/s")
            log_writer.writerow([iter_num, tokens_seen, f"{loss_accum:.4f}", "", f"{lr:.3e}",
                                 f"{tok_s:.0f}", f"{now - t_start:.0f}"])
            log_file.flush()

        # ---- time limit ----
        if iter_num % 10 == 0 and should_stop_for_time():
            if master:
                print(f"time limit {time_limit_hours} h reached at step {iter_num}; saving and exiting")
                save_ckpt(os.path.join(ckpt_dir, "ckpt_latest.pt"), None)
            break

except KeyboardInterrupt:
    if master:
        print(f"interrupted at step {iter_num}; saving ckpt_latest.pt")
        save_ckpt(os.path.join(ckpt_dir, "ckpt_latest.pt"), None)

if master:
    log_file.close()
    print(f"done. steps={iter_num} tokens={tokens_seen/1e6:.1f}M best_val={best_val_loss:.4f} "
          f"elapsed={(time.time()-t_start)/60:.1f} min")
if ddp:
    destroy_process_group()
