"""C01 v1 reference decoder."""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int
    tokenizer_id: str
    architecture_version: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_sequence_length: int
    rope_theta: float
    rms_norm_eps: float
    initializer_std: float

    def __post_init__(self):
        fixed = dict(architecture_version=1, num_layers=20, hidden_size=2048,
                     num_attention_heads=16, num_key_value_heads=4, intermediate_size=5632,
                     max_sequence_length=4096, rope_theta=10000., rms_norm_eps=1e-5,
                     initializer_std=0.02)
        for name, expected in fixed.items():
            value = getattr(self, name)
            types = (int,) if type(expected) is int else (float, int)
            if type(value) not in types or value != expected:
                raise ValueError(f"Invalid {name}: expected {expected}")
        if type(self.vocab_size) is not int or self.vocab_size < 1:
            raise ValueError("Expected positive integer vocab_size")
        if not isinstance(self.tokenizer_id, str) or not self.tokenizer_id.strip():
            raise ValueError("Expected nonempty tokenizer_id")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.rms_norm_eps

    def forward(self, x):
        z = x.float()
        return (self.weight.float() * z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)


def rope_angles(length, dim, theta, device):
    index = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freq = theta ** (-index / dim)
    angles = torch.arange(length, device=device)[:, None, None] * freq
    return angles.cos(), angles.sin()


def rotate(x, cos, sin):
    even, odd = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((even*cos - odd*sin, even*sin + odd*cos), dim=-1).flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dim = c.hidden_size // c.num_attention_heads
        self.group = c.num_attention_heads // c.num_key_value_heads
        kv = self.dim * c.num_key_value_heads
        self.q = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.k = nn.Linear(c.hidden_size, kv, bias=False)
        self.v = nn.Linear(c.hidden_size, kv, bias=False)
        self.o = nn.Linear(c.hidden_size, c.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask):
        q = rotate(self.q(x).unflatten(-1, (-1, self.dim)), cos, sin).transpose(1, 2)
        k = rotate(self.k(x).unflatten(-1, (-1, self.dim)), cos, sin).transpose(1, 2)
        v = self.v(x).unflatten(-1, (-1, self.dim)).transpose(1, 2)
        # Right padding is in the causal future of every valid query.
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = F.scaled_dot_product_attention(
                q.float(), k.float().repeat_interleave(self.group, 1),
                v.float().repeat_interleave(self.group, 1), is_causal=True)
        out = self.o(out.to(v.dtype).transpose(1, 2).flatten(-2))
        return out if mask is None else out.masked_fill(~mask[..., None], 0)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(c), RMSNorm(c)
        self.attn = Attention(c)
        self.gate = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        z = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config)
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, std=config.initializer_std)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected int32 or int64 input_ids[B,T]")
        b, t = input_ids.shape
        if b < 1 or not 1 <= t <= self.config.max_sequence_length:
            raise ValueError("Invalid batch or sequence length")
        if ((input_ids < 0) | (input_ids >= self.config.vocab_size)).any():
            raise ValueError("Token outside vocabulary")
        if attention_mask is not None:
            m = attention_mask
            if m.shape != input_ids.shape or m.dtype != torch.bool or m.device != input_ids.device:
                raise ValueError("Invalid attention_mask")
            if not m[:, 0].all() or (m[:, 1:] & ~m[:, :-1]).any():
                raise ValueError("Expected nonempty right-padded prefixes")
        x = self.embedding(input_ids)
        dim = self.config.hidden_size // self.config.num_attention_heads
        cos, sin = rope_angles(t, dim, self.config.rope_theta, x.device)
        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)
        logits = F.linear(self.norm(x), self.embedding.weight)
        return logits if attention_mask is None else logits.masked_fill(~attention_mask[..., None], 0)
