"""Reproduce the production-shape export check; needs 7.3 GiB free disk."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.c07_export import Config, Decoder, digest, export

ROOT = Path(__file__).resolve().parents[1]


def check(work):
    work.mkdir(parents=True, exist_ok=False)
    token_dir = ROOT / "artifacts/c02-v1"
    token_meta = json.loads((token_dir / "manifest.json").read_bytes())
    cfg = json.loads((ROOT / "configs/c03-fixture.json").read_bytes())
    cfg.update({k: token_meta[k] for k in ("tokenizer_id", "vocab_size")})
    config = Config(**cfg)
    with torch.device("meta"):
        shapes = Decoder(config).state_dict()
    # A distinct repeated row per key keeps the source checkpoint small;
    # the output still contains every dense FP32 element.
    state = {name: (torch.arange(v.shape[-1], dtype=torch.float32) + 10000*i).expand(v.shape)
             for i, (name, v) in enumerate(sorted(shapes.items()))}
    for mode in ("c03", "c05"):
        checkpoint = (dict(config=asdict(config), state_dict=state) if mode == "c03" else
                      dict(signature={"config": {"model": asdict(config)}}, model=state,
                           optimizer={"sentinel": torch.tensor(123)}))
        path = work / f"{mode}.pt"
        torch.save(checkpoint, path)
        export(path, token_dir, work / mode)
    m = (work/"c03/manifest.json").read_bytes()
    assert m == (work/"c05/manifest.json").read_bytes()
    for name, info in json.loads(m)["files"].items():
        for mode in ("c03", "c05"):
            path = work/mode/name
            assert path.stat().st_size == info["bytes"] and digest(path) == info["sha256"]
    with safe_open(work/"c03/model.safetensors", framework="pt") as weights:
        assert set(weights.keys()) == set(state)
        for name, expected in state.items():
            actual = weights.get_tensor(name)
            assert actual.dtype == torch.float32 and torch.equal(actual, expected), name
    print("Two reproducible exports, tensors:", len(state),
          "weight SHA-256:", json.loads(m)["files"]["model.safetensors"]["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    check(parser.parse_args().work)
