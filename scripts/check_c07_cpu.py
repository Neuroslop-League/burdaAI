"""Build a synthetic C01 bundle and compare four Rust CPU forwards with PyTorch."""
import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from burdaai.decoder import Config, Decoder
from scripts.c07_export import export

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Привет, мир!"


def reference(work):
    work.mkdir(parents=True, exist_ok=False)
    td = ROOT / "artifacts/c02-v1"
    tm = json.loads((td / "manifest.json").read_bytes())
    cfg = json.loads((ROOT / "configs/c03-fixture.json").read_bytes())
    cfg.update({k: tm[k] for k in ("tokenizer_id", "vocab_size")})
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = Decoder(Config(**cfg)).eval()
    checkpoint = work / "synthetic.pt"
    torch.save(dict(config=asdict(model.config), state_dict=model.state_dict()), checkpoint)
    del model
    bundle = work / "bundle"
    export(checkpoint, td, bundle)
    checkpoint.unlink()
    with torch.device("meta"):
        model = Decoder(Config(**cfg))
    model.load_state_dict(load_file(bundle / "model.safetensors"), assign=True)
    model.eval()
    tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    ids = [1, *tokenizer.encode(PROMPT, add_special_tokens=False).ids]
    print("prompt:", ids, flush=True)
    with torch.inference_mode():
        for step in range(4):
            logits = model(torch.tensor([ids]))[0].numpy()
            assert np.isfinite(logits).all()
            np.save(work/f"python-{step}.npy", logits)
            ids.append(int(logits[-1].argmax()))
            assert ids[-1] != 2, "fixture ended before four steps"


def compare(work, binary):
    trace = work / "rust-logits"
    subprocess.run([binary, "--bundle",work/"bundle","--text",PROMPT,"--device","cpu",
                    "--placement","cpu","--max-new-tokens","4","--trace-dir",trace],check=True)
    if sys.platform == "darwin":
        import resource
        print("Rust peak RSS bytes:", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    with (trace/"steps.json").open() as f:
        steps = json.load(f)
    assert len(steps) == 4
    for i, step in enumerate(steps):
        ref = np.load(work/f"python-{i}.npy")
        assert step["shape"] == list(ref.shape)
        other = np.fromfile(trace/f"{i}.f32",dtype="<f4").reshape(ref.shape)
        error = np.abs(other-ref).max()
        expected = int(ref[-1].argmax())
        assert step["id"] == expected and error <= 1e-3, (i,expected,step,error)
        print(i, ref.shape, expected, step["id"], float(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["reference","compare"])
    parser.add_argument("work",type=Path)
    parser.add_argument("--binary",type=Path,default=ROOT/"rust/target/release/burda-generate")
    args = parser.parse_args()
    if args.phase == "reference":
        reference(args.work)
    else:
        compare(args.work,args.binary)
