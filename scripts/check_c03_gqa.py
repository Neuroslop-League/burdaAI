"""Independent FP32 attention oracle and decoder prefix check. Run as a module."""
import math
from pathlib import Path

import torch

from burdaai.decoder import Attention, Config, Decoder, rope_angles, rotate


def reference_rotation(heads, theta):
    result = torch.empty_like(heads)
    for pos in range(heads.size(1)):
        for i in range(0, heads.size(-1), 2):
            angle = pos / theta ** (i / heads.size(-1))
            a, b = heads[:, pos, :, i], heads[:, pos, :, i + 1]
            result[:, pos, :, i] = a * math.cos(angle) - b * math.sin(angle)
            result[:, pos, :, i + 1] = a * math.sin(angle) + b * math.cos(angle)
    return result


def reference(q, k, v, mask, theta):
    q, k = reference_rotation(q, theta), reference_rotation(k, theta)
    out = torch.zeros_like(q)
    for batch in range(len(q)):
        for pos in range(int(mask[batch].sum())):
            for head in range(16):
                group = head // 4
                keys = k[batch, :pos + 1, group]
                scores = (keys @ q[batch, pos, head]) / math.sqrt(q.size(-1))
                out[batch, pos, head] = scores.softmax(0) @ v[batch, :pos + 1, group]
    return out


def compare(label, actual, expected, tolerance):
    error = (actual - expected).abs().max().item()
    print(f"{label}: {error:.9g} (limit {tolerance:g})")
    if not error <= tolerance:
        raise AssertionError(label)


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs/c03-fixture.json")
    attn = Attention(cfg)
    # Identity O exposes individual heads without mixing or round-off.
    attn.o.weight.copy_(torch.eye(cfg.hidden_size))
    x = torch.randn(2, 5, cfg.hidden_size)
    mask = torch.tensor([[True]*5, [True]*3 + [False]*2])
    q = attn.q(x).reshape(2, 5, 16, 128)
    k = attn.k(x).reshape(2, 5, 4, 128)
    v = attn.v(x).reshape_as(k)
    assert v[0, 0].unique(dim=0).size(0) == 4
    cos, sin = rope_angles(5, 128, cfg.rope_theta, x.device)
    for label, heads in (("Q", q), ("K", k)):
        compare("RoPE position 0 " + label, rotate(heads, cos, sin)[:, 0], heads[:, 0], 0)
    actual = attn(x, cos, sin, mask).reshape_as(q)
    for group in range(4):
        compare(f"KV {group}, Q {4*group}..{4*group+3}", actual[:, 0, 4*group:4*group+4],
                v[:, 0, group:group+1], 0)
    compare("GQA vs FP32", actual, reference(q, k, v, mask, cfg.rope_theta), 1e-5)

    model = Decoder(cfg).eval()
    # Only the first three positions share the same input tokens.
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]])
    logits = model(ids)
    compare("common prefix logits", logits[0, :3], logits[1, :3], 1e-5)


if __name__ == "__main__":
    main()
