"""A small, GPT-style decoder-only Transformer for Python code experiments."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    dim: int = 512
    max_seq_len: int = 1024
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    def to_dict(self): return asdict(self)


class CausalSelfAttention(nn.Module):
    """Project Q/K/V, mask future positions, attend, then project back."""
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_heads == 0
        self.heads, self.head_dim = config.n_heads, config.dim // config.n_heads
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.out = nn.Linear(config.dim, config.dim)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        # is_causal lets SDPA pick its fastest kernel (Flash/memory-efficient) without
        # building and storing a max_seq_len x max_seq_len mask.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_drop(self.out(y))


class MLP(nn.Module):
    """The position-wise 4x expansion, GELU nonlinearity, and projection."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(config.dim, 4 * config.dim), nn.GELU(), nn.Linear(4 * config.dim, config.dim), nn.Dropout(config.dropout))
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-normalized attention and MLP, each joined by a residual path."""
    def __init__(self, config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(config.dim), nn.LayerNorm(config.dim)
        self.attn, self.mlp = CausalSelfAttention(config), MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class DecoderOnlyTransformer(nn.Module):
    """Token/position embeddings, stacked decoder blocks, and tied LM head."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.dim)
        self.position_embedding = nn.Embedding(c.max_seq_len, c.dim)
        self.embedding_dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.dim)
        self.lm_head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying saves memory and improves token learning.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        b, t = input_ids.shape
        if t > self.config.max_seq_len: raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(t, device=input_ids.device)
        x = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks: x = block(x)
        logits = self.lm_head(self.final_norm(x))
        # Inputs contain a context window; targets are its one-token-shifted answers.
        # Targets of -100 (e.g. the user's lines in chat data) are not trained on. Averaging
        # over the real targets keeps a window with none of them at a loss of 0, not NaN.
        loss = None
        if targets is not None:
            flat = targets.reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), flat, ignore_index=-100, reduction="sum")
            loss = loss / (flat != -100).sum().clamp(min=1)
        return logits, loss


# Friendly short alias for scripts and notebooks.
GPT = DecoderOnlyTransformer
