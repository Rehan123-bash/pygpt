# Main run: GPT-2 Small architecture (12L / 12H / 768d) on Python code, Kaggle 2x T4.
# Usage (Kaggle): torchrun --standalone --nproc_per_node=2 train.py --config config/gpt2_small_py.py
#
# Two values MUST be set after the smoke test on real hardware:
#   micro_batch_size  - largest value that fits in 16 GB (try 16, then 12, then 8)
#   max_iters         - (measured tokens/sec * 36000) // tokens_per_step
#                       where tokens_per_step = micro_batch_size * block_size * n_gpus * grad_accum_steps

# model
n_layer = 12
n_head = 12
n_embd = 768
block_size = 512
vocab_size = 32768               # codeparrot/codeparrot tokenizer; fits uint16 exactly
dropout = 0.0
bias = False

# data
data_dir = "/kaggle/input/pygpt-python-tokens"
micro_batch_size = 16            # per GPU; set from smoke test
grad_accum_steps = 8             # per GPU; tokens/step = 16 * 512 * 2 GPUs * 8 = 131,072

# optimizer / schedule
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 200
max_iters = 3800                 # ~500M tokens at 131k tokens/step; set from measured throughput
beta1, beta2 = 0.9, 0.95
weight_decay = 0.1
grad_clip = 1.0

# system
dtype = "float16"                # T4 has no bf16
compile = False                  # flip to True only if the smoke test passes with it on T4
eval_interval = 250
eval_iters = 40
log_interval = 10
ckpt_dir = "/kaggle/working"
time_limit_hours = 10.0          # save + exit cleanly before Kaggle's session cap
seed = 1337
