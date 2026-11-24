"""CPU preflight and explicit CUDA training entry points for the C04 pilot."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from burdaai.decoder import Config, Decoder
from burdaai.offload import HostAdamW
from scripts.c02_artifact import load

ROOT = Path(__file__).resolve().parents[1]


def read_inputs(cfg):
    model = Config(**cfg["model"])
    path = ROOT / cfg["manifest"]
    manifest = json.loads(path.read_bytes())
    directory = ROOT / cfg["tokenizer"]
    tokenizer = load(directory)
    meta = json.loads((directory / "manifest.json").read_bytes())
    for field in ("tokenizer_id", "vocab_size"):
        if manifest[field] != meta[field] or manifest[field] != getattr(model, field):
            raise ValueError(f"Incompatible {field}")
    if manifest["format_version"] != 1 or tokenizer.get_vocab_size() != model.vocab_size:
        raise ValueError("Incompatible data format or vocabulary")
    splits = {}
    seen = set()
    for split in ("train", "validation", "test"):
        info = manifest["splits"][split]
        raw = (path.parent / info["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != info["sha256"] or len(raw) != info["bytes"]:
            raise ValueError(f"Corrupt {split} shard")
        rows = [json.loads(s) for s in raw.splitlines()]
        ids = [r["input_ids"] for r in rows]
        if len(rows) != info["windows"] or sum(map(len, ids)) != info["stored_ids"]:
            raise ValueError(f"Wrong {split} counts")
        docs = {r["document_id"] for r in rows}
        if seen & docs:
            raise ValueError("Document crosses splits")
        seen.update(docs)
        for seq in ids:
            if not 2 <= len(seq) <= model.max_sequence_length or any(
                type(t) is not int or not 0 <= t < model.vocab_size for t in seq
            ):
                raise ValueError("Invalid token row")
        splits[split] = ids
    return splits, hashlib.sha256(path.read_bytes()).hexdigest()


def windows(rows, length):
    return [row[i:i+length] for row in rows for i in range(0, len(row)-1, length-1)]


def batch(rows, device="cpu"):
    size = max(map(len, rows))
    x = torch.tensor([row + [0]*(size-len(row)) for row in rows], device=device)
    mask = torch.arange(size, device=device) < torch.tensor(list(map(len, rows)), device=device)[:, None]
    targets = torch.full_like(x, -100)
    targets[:, :-1] = torch.where(mask[:, 1:], x[:, 1:], -100)
    return x, mask, targets


def loss_sum(logits, targets):
    return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="sum",
                           ignore_index=-100)


def preflight(splits):
    for name, rows in splits.items():
        x, mask, y = batch(rows)
        assert (x[~mask] == 0).all() and (y != -100).sum() == sum(len(r)-1 for r in rows)
        assert torch.equal(y[y != -100], x[:, 1:][mask[:, 1:]])
        assert sum(len(w)-1 for w in windows(rows, 32)) == sum(len(r)-1 for r in rows)
        print(name, tuple(x.shape), "valid targets", int((y != -100).sum()))
    x, mask, y = batch([[1, 3, 2], [1, 2]])
    assert x.tolist() == [[1,3,2],[1,2,0]] and mask.tolist() == [[True]*3,[True,True,False]]
    assert y.tolist() == [[3,2,-100],[2,-100,-100]]
    logits = torch.zeros(2,3,4)
    logits[0,0,3] = math.log(3)
    logits[1,0,2] = math.log(2)
    logits[y == -100] = 123
    actual = loss_sum(logits, y).item()/3
    expected = (math.log(2)+math.log(4)+math.log(2.5))/3
    assert abs(actual-expected) < 1e-6
    print(f"CPU FP32 loss {actual:.9f}, analytic {expected:.9f}, abs {abs(actual-expected):.3g}")


def check_offload():
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    saved = p.detach().clone()
    opt = HostAdamW([p], lr=1e-4)
    host = opt.host_params[0]
    assert host.data_ptr() != p.data_ptr()
    p.grad = torch.tensor([.25, -.5])
    opt.stage_gradients()
    assert p.grad is None and torch.equal(host.grad, torch.tensor([.25, -.5]))
    # Synthetic completed-step state exercises the actual resume path without stepping.
    sample = dict(step=torch.tensor(1.), exp_avg=torch.tensor([.1, -.2]),
                  exp_avg_sq=torch.tensor([.01, .04]))
    opt.state[host] = sample
    with tempfile.TemporaryFile() as file:
        torch.save(opt.state_dict(), file)
        file.seek(0)
        restored = HostAdamW([p], lr=1e-4)
        restored.load_state_dict(torch.load(file, weights_only=True, map_location="cpu"))
    for name, value in restored.state[restored.host_params[0]].items():
        assert value.device.type == "cpu" and value.dtype == torch.float32
        assert torch.equal(value, sample[name])
    # Evaluate the first-step AdamW formula on scratch tensors only.
    gradient = host.grad
    moment = .1 * gradient
    variance = .001 * gradient.square()
    direction = (moment / .1) / ((variance / .001).sqrt() + 1e-8)
    torch.testing.assert_close(direction, gradient.sign())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        probe = torch.eye(4) @ torch.eye(4)
    assert probe.dtype == torch.bfloat16 and torch.equal(probe.float(), torch.eye(4))
    opt.zero_grad()
    assert host.grad is None and torch.equal(host, saved) and torch.equal(p, saved)
    print("CPU offload: independent FP32 master, gradient staging, FP32 checkpoint round-trip OK")


def peak_ram():
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss*1024


def train(args, cfg, splits, digest):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native CUDA BF16 required for this profile")
    if not 2 <= args.length <= 4096 or min(args.microbatch,args.accumulate,cfg["checkpoint_every"],cfg["steps"]) < 1:
        raise ValueError("Invalid training dimensions or interval")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "latest.pt"
    if path.exists() != args.resume:
        raise ValueError("Use a fresh output directory or --resume with latest.pt")
    torch.manual_seed(cfg["seed"])
    signature = dict(config=cfg, manifest_sha256=digest, length=args.length, microbatch=args.microbatch,
                     accumulate=args.accumulate, torch=str(torch.__version__), cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name())
    model = Decoder(Config(**cfg["model"])).cuda()
    if cfg["optimizer_device"] != "cpu":
        raise ValueError("RTX 5070 profile requires CPU optimizer")
    opt = HostAdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                    eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    rows = windows(splits["train"], args.length)
    random.Random(cfg["seed"]).shuffle(rows)
    start = cursor = tokens = 0
    if args.resume:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if state["signature"] != signature:
            raise ValueError("Checkpoint profile mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start, cursor, tokens = state["step"], state["cursor"], state["tokens"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state
    for step in range(start+1, cfg["steps"]+1):
        selected = [rows[(cursor+i)%len(rows)] for i in range(args.accumulate*args.microbatch)]
        count = sum(len(r)-1 for r in selected)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        summed = 0.
        for i in range(0,len(selected),args.microbatch):
            x, mask, y = batch(selected[i:i+args.microbatch], "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, mask)
            loss = loss_sum(logits, y)
            if not loss.isfinite():
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            (loss/count).backward()
            summed += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-before
        if any(t.dtype != torch.float32 or t.device.type != "cpu" for st in opt.state.values() for t in st.values() if torch.is_tensor(t)):
            raise ValueError("AdamW state must be FP32")
        cursor += len(selected)
        tokens += count
        print(json.dumps(dict(step=step,loss=summed/count,targets=count,lr=cfg["lr"],seconds=elapsed,
                              tokens_per_second=count/elapsed,vram_allocated=torch.cuda.max_memory_allocated(),
                              vram_reserved=torch.cuda.max_memory_reserved(),peak_ram_bytes=peak_ram())), flush=True)
        if step % cfg["checkpoint_every"] == 0 or step == cfg["steps"]:
            with tempfile.NamedTemporaryFile(dir=args.output,delete=False) as temp:
                tmp = Path(temp.name)
            try:
                torch.save(dict(signature=signature, model=model.state_dict(),optimizer=opt.state_dict(),
                                step=step,cursor=cursor,tokens=tokens,rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state()),tmp)
                check = torch.load(tmp,weights_only=True,map_location="meta",mmap=True)
                if check["step"] != step:
                    raise ValueError("Invalid checkpoint step")
                del check
                os.replace(tmp,path)
            finally:
                tmp.unlink(missing_ok=True)
            print("checkpoint",path,"peak RAM bytes",peak_ram(),flush=True)
    total = valid = 0
    model.eval()
    with torch.inference_mode():
        for row in windows(splits["validation"], args.length):
            x, mask, y = batch([row],"cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits = model(x,mask)
            total += loss_sum(logits,y).item()
            valid += len(row)-1
    print("validation loss",total/valid,"targets",valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check")
    run = modes.add_parser("train")
    run.add_argument("--length", type=int, required=True)
    run.add_argument("--microbatch", type=int, required=True)
    run.add_argument("--accumulate", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT/"configs/c05-pilot.json").read_bytes())
    splits, digest = read_inputs(cfg)
    if args.mode == "check":
        preflight(splits)
        check_offload()
    else:
        train(args,cfg,splits,digest)
