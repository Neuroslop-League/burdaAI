"""Export an FP32 C03 decoder or C05 training checkpoint into a C07.1 bundle."""
import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from burdaai.decoder import Config, Decoder


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_bytes((json.dumps(data, sort_keys=True, indent=2) + "\n").encode())


def write_weights(path, state, config):
    with torch.device("meta"):
        expected = Decoder(config).state_dict()
    if state.keys() != expected.keys():
        raise ValueError("Missing or unexpected state_dict keys")
    header = {}
    offset = 0
    for name in sorted(state):
        tensor = state[name]
        if tensor.shape != expected[name].shape or tensor.dtype != torch.float32:
            raise ValueError(f"Wrong shape or dtype: {name}")
        end = offset + tensor.numel() * 4
        header[name] = dict(dtype="F32", shape=list(tensor.shape), data_offsets=[offset, end])
        offset = end
    # Safetensors v1 wire format; stream one contiguous tensor at a time.
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for name in header:
            array = state[name].detach().contiguous().numpy().astype("<f4", copy=False)
            stream.write(memoryview(array).cast("B"))


def export(checkpoint, tokenizer_dir, output):
    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=True)
    if "state_dict" in saved:
        config, state = Config(**saved["config"]), saved["state_dict"]
    else:
        config, state = Config(**saved["signature"]["config"]["model"]), saved["model"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        shutil.copyfile(tokenizer_dir / "tokenizer.json", bundle / "tokenizer.json")
        shutil.copyfile(tokenizer_dir / "manifest.json", bundle / "tokenizer-manifest.json")
        meta = json.loads((bundle / "tokenizer-manifest.json").read_bytes())
        tok_path = bundle / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tok_path))
        sha = digest(tok_path)
        if (meta["file"] != "tokenizer.json" or sha != meta["sha256"]
                or meta["tokenizer_id"] != config.tokenizer_id
                or config.tokenizer_id != "burda-bpe-v1-sha256-" + sha
                or meta["vocab_size"] != config.vocab_size
                or sorted(tokenizer.get_vocab().values()) != list(range(config.vocab_size))
                or meta["special_tokens"] != {"PAD": 0, "BOS": 1, "EOS": 2}
                or [tokenizer.token_to_id(s) for s in ("<|pad|>", "<|bos|>", "<|eos|>")] != [0, 1, 2]):
            raise ValueError("Tokenizer incompatible with checkpoint")
        write_json(bundle / "config.json", asdict(config))
        write_weights(bundle / "model.safetensors", state, config)
        files = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in bundle.iterdir()}
        write_json(bundle / "manifest.json", dict(schema_version=1, profile="reference-f32",
                   tokenizer_id=config.tokenizer_id, files=files))
        bundle.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/c02-v1"))
    args = parser.parse_args()
    export(args.checkpoint, args.tokenizer, args.output)
