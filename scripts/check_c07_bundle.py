"""Verify published C07.1 fixture and compare the layout with C01 Python."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from burdaai.decoder import Config, Decoder

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "fixtures/c07-v1"


def layout(c):
    h, f = c.hidden_size, c.intermediate_size
    kv = c.num_key_value_heads * (h // c.num_attention_heads)
    shapes = {"embedding.weight": (c.vocab_size, h), "norm.weight": (h,)}
    for i in range(c.num_layers):
        for name, shape in {"attn_norm": (h,), "ffn_norm": (h,), "attn.q": (h,h),
                            "attn.k": (kv,h), "attn.v": (kv,h), "attn.o": (h,h),
                            "gate": (f,h), "up": (f,h), "down": (h,f)}.items():
            shapes[f"blocks.{i}.{name}.weight"] = shape
    return shapes


def main():
    config = json.loads((ROOT/"configs/c03-fixture.json").read_bytes())
    meta = json.loads((ROOT/"artifacts/c02-v1/manifest.json").read_bytes())
    config.update({k: meta[k] for k in ("tokenizer_id","vocab_size")})
    prod = Config(**config)
    with torch.device("meta"):
        state = Decoder(prod).state_dict()
    assert {k: tuple(v.shape) for k,v in state.items()} == layout(prod)
    assert all(v.dtype == torch.float32 for v in state.values())
    print("C01 tensors:",len(state),"elements:",sum(v.numel() for v in state.values()))

    manifest = json.loads((BUNDLE/"manifest.json").read_bytes())
    assert (manifest["schema_version"],manifest["profile"]) == (1,"loader-fixture")
    for name, expected in manifest["files"].items():
        data = (BUNDLE/name).read_bytes()
        assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"]
    c = SimpleNamespace(**json.loads((BUNDLE/"config.json").read_bytes()))
    tm = json.loads((BUNDLE/"tokenizer-manifest.json").read_bytes())
    assert manifest["tokenizer_id"] == c.tokenizer_id == tm["tokenizer_id"]
    assert tm["sha256"] == manifest["files"]["tokenizer.json"]["sha256"]
    tok = json.loads((BUNDLE/"tokenizer.json").read_bytes())["model"]["vocab"]
    assert sorted(tok.values()) == list(range(c.vocab_size)) and c.vocab_size == tm["vocab_size"]
    assert [tok[s] for s in ("<pad>", "<bos>", "<eos>")] == [tm["special_tokens"][r] for r in ("PAD","BOS","EOS")]
    weights = load_file(BUNDLE/"model.safetensors")
    assert {k: tuple(v.shape) for k,v in weights.items()} == layout(c)
    for i, name in enumerate(sorted(weights)):
        v = weights[name]
        assert v.dtype == torch.float32
        assert torch.equal(v.flatten(), torch.arange(v.numel(),dtype=torch.float32) + 1000*i)
    print("Fixture tensors:",len(weights),"elements:",sum(v.numel() for v in weights.values()))


if __name__ == "__main__":
    main()
