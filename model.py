"""
model.py - A GPT-2-style decoder-only transformer, written from scratch in PyTorch.

Architecture (GPT-2 Small when n_layer=12, n_head=12, n_embd=768):

    tokens --> token embedding + position embedding
           --> N x [ x + Attn(LN(x)) ; x + MLP(LN(x)) ]      (pre-LayerNorm blocks)
           --> LayerNorm --> linear head (tied to token embedding) --> logits

Design references: Vaswani et al. 2017 (transformer), Radford et al. 2019 (GPT-2),
Press & Wolf 2017 (weight tying), Karpathy's nanoGPT (code structure).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32768      # codeparrot tokenizer
    block_size: int = 512        # max context length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False           # no bias in Linear/LayerNorm: slightly faster, same quality


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (PyTorch's doesn't support bias=False everywhere)."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask: position t can only attend to
    positions <= t. This is what makes next-token prediction well-defined.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # one fused projection for q, k, v (3 * n_embd outputs)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection back to the residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # PyTorch >= 2.0 has a fused kernel (flash on Ampere+, memory-efficient on T4)
        self.use_sdpa = hasattr(F, "scaled_dot_product_attention")
        if not self.use_sdpa:
            # fallback: explicit lower-triangular mask
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.shape                                   # batch, time, channels
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))     # (B, nh, T, T)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v                                                 # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)   # re-assemble heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, GELU, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),   # learned position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output projection reuses the input embedding matrix.
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 initialisation
        self.apply(self._init_weights)
        # Scale the residual-stream projections by 1/sqrt(2 * n_layer) so the sum of
        # 2*n_layer residual contributions keeps unit variance.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token ids
        targets: (B, T) next-token ids, or None at inference
        returns: logits (B, T, vocab), loss (scalar or None)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # cross-entropy over every position at once; ignore_index=-1 lets callers mask
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference: only the last position's logits are needed
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        AdamW with decoupled weight decay applied only to >=2-D tensors (weight matrices,
        embeddings). Biases and LayerNorm gains are not decayed.
        """
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in params.items() if p.dim() >= 2]
        no_decay = [p for n, p in params.items() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # fused AdamW is faster on CUDA when available
        import inspect
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        extra = {"fused": True} if fused_ok else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_token_id=None):
        """
        Autoregressive sampling.
          temperature -> 0: greedy. Higher = more random.
          top_k:  keep only the k most likely tokens.
          top_p:  nucleus sampling (Holtzman et al. 2020): keep the smallest set of tokens
                  whose cumulative probability >= p.
          stop_token_id: stop early when this id is produced (e.g. <|endoftext|>).
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()

            if temperature <= 1e-6:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum - F.softmax(sorted_logits, dim=-1) > top_p   # keep the first token that crosses p
                    sorted_logits[remove] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            if stop_token_id is not None and (next_id == stop_token_id).all():
                break
        return idx


if __name__ == "__main__":
    # quick self-check: parameter count for the main config
    cfg = GPTConfig()
    m = GPT(cfg)
    print(f"GPT-2 Small architecture with vocab {cfg.vocab_size}, ctx {cfg.block_size}: "
          f"{m.num_params()/1e6:.1f}M params ({m.num_params(non_embedding=True)/1e6:.1f}M excl. position emb)")
