# Tiny config for bug-finding. Runs in ~1 minute on CPU, seconds on GPU.
# Usage: python train.py --config config/smoke.py --data_dir data/synthetic

# model
n_layer = 4
n_head = 4
n_embd = 128
block_size = 128
vocab_size = 32768
dropout = 0.0
bias = False

# data
data_dir = "data/synthetic"      # created by tests/make_synthetic_data.py
micro_batch_size = 8
grad_accum_steps = 1

# optimizer / schedule
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 10
max_iters = 60
beta1, beta2 = 0.9, 0.95
weight_decay = 0.1
grad_clip = 1.0

# system
dtype = "float16"                # auto-downgrades to float32 on CPU
compile = False
eval_interval = 20
eval_iters = 5
log_interval = 5
ckpt_dir = "runs/smoke"
time_limit_hours = 1.0
seed = 1337
