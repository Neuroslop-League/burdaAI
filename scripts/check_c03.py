"""Run from repository root: python -m scripts.check_c03."""
from dataclasses import asdict
from pathlib import Path

import torch

from burdaai.decoder import Config, Decoder


def check():
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs/c03-fixture.json")
    fields = asdict(cfg)
    for name in fields:
        for invalid in ({k: v for k, v in fields.items() if k != name}, fields | {name: None}):
            try:
                Config(**invalid)
            except (TypeError, ValueError):
                continue
            raise AssertionError(name)
    for overrides in (dict(num_layers=19), dict(architecture_version=2), dict(vocab_size=0)):
        try:
            Config(**(fields | overrides))
        except ValueError:
            continue
        raise AssertionError(overrides)

    torch.manual_seed(42)
    torch.set_default_dtype(torch.bfloat16)
    model = Decoder(cfg)
    count = sum(p.numel() for p in model.parameters())
    assert len(model.blocks) == 20
    assert count == 20 * (10485760 + 34603008 + 4096) + 2048 + 8 * 2048
    # Only E needs a gradient for this check on an 8 GiB CPU host.
    model.requires_grad_(False)
    model.embedding.weight.requires_grad_(True)
    shapes = {}

    def record(name):
        def hook(module, inputs, output):
            shapes[name] = tuple(output.shape)
        return hook

    block = model.blocks[0]
    for name in ("attn.q", "attn.k", "attn.v", "attn.o", "gate", "up", "down"):
        block.get_submodule(name).register_forward_hook(record(name))
    ids = torch.tensor([[1, 2, 3]])
    logits = model(ids)
    expected = {"attn.q": 2048, "attn.k": 512, "attn.v": 512, "attn.o": 2048,
                "gate": 5632, "up": 5632, "down": 2048}
    assert shapes == {name: (1, 3, width) for name, width in expected.items()}
    assert logits.shape == (1, 3, 8) and logits.isfinite().all()
    logits.float().square().mean().backward()
    grad = model.embedding.weight.grad
    assert grad is not None and grad.isfinite().all() and grad.abs().max() > 0
    print(torch.__version__, "CPU BF16, blocks:", len(model.blocks), "parameters:", count)
    print(shapes, "logits:", tuple(logits.shape), "shared E gradient:", tuple(grad.shape))


if __name__ == "__main__":
    check()
