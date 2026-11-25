"""C01.4 runner on a single Linux AMD Vulkan card exposing amdgpu VRAM counters."""
import argparse
import hashlib
import json
import math
import resource
import subprocess
import tempfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def bench(args):
    cfg=json.loads((ROOT/"configs/c07-1gib.json").read_bytes())
    if (cfg["backend"],cfg["format"],cfg["prompt_ids"],cfg["steps"],cfg["ignore_eos"]) != ("vulkan","Q4_0",256,64,True):
        raise ValueError("Unsupported C01.4 acceptance profile")
    dev=args.drm.resolve()/"device"
    total = dev/"mem_info_vram_total"
    if not total.is_file() or not (dev/"mem_info_vram_used").is_file():
        raise ValueError(f"{args.drm}: amdgpu VRAM counters missing; check Linux DRM card and driver")
    if int(total.read_text()) != cfg["vram_bytes"]:
        raise ValueError("C01.4 requires a dedicated 1 GiB device")
    # Correlate DRM telemetry and Vulkan by allowing only one discrete AMD adapter.
    cards=[p for p in Path("/sys/class/drm").glob("card[0-9]*") if p.name[4:].isdigit()
           and (p/"device/mem_info_vram_total").exists()]
    if len(cards)!=1 or cards[0].resolve()!=args.drm.resolve():
        raise ValueError("Select the sole amdgpu adapter")
    used=dev/"mem_info_vram_used"
    base=int(used.read_text())
    peak_vram=base
    peak_ram=0
    meta=json.loads((ROOT/"artifacts/c02-v1/manifest.json").read_bytes())
    ids=[1]+[3+(i*997)%32765 for i in range(cfg["prompt_ids"]-1)]
    with args.model.open("rb") as f:
        model_hash=hashlib.file_digest(f,"sha256").hexdigest()
    with tempfile.TemporaryFile(mode="w+") as out, args.log.open("x") as log:
        cmd=[args.runner,args.model,str(cfg["gpu_layers"]),str(cfg["steps"]),meta["tokenizer_id"],
             ",".join(map(str,ids)),"benchmark"]
        with subprocess.Popen(cmd,stdout=out,stderr=log) as proc:
            monitor=psutil.Process(proc.pid)
            while proc.poll() is None:
                peak_vram=max(peak_vram,int(used.read_text()))
                try:
                    peak_ram=max(peak_ram,monitor.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                time.sleep(cfg["poll_ms"]/1000)
            if proc.wait()!=0:
                raise RuntimeError(f"backend failed; see {args.log}")
        out.seek(0)
        generated=[int(s) for s in out]
    lines=args.log.read_text().splitlines()
    proof=[s for s in lines if s.startswith("completed_gpu_matmul=")]
    timings=[s for s in lines if s.startswith("prompt=")]
    if len(generated)!=cfg["steps"] or not proof or not timings or peak_ram==0:
        raise ValueError("Incomplete acceptance evidence")
    fields = dict(item.split("=",1) for item in timings[-1].split())
    if int(fields["prompt"]) != cfg["prompt_ids"] or int(fields["emitted"]) != cfg["steps"]:
        raise ValueError("Backend reported unexpected scenario dimensions")
    metrics = {k: float(fields[k]) for k in ("cold_ttft_s","warm_ttft_s","tokens_per_second_after_first")}
    if not all(math.isfinite(v) and v > 0 for v in metrics.values()):
        raise ValueError("Invalid backend timing")
    print(json.dumps(dict(metrics=metrics, model_sha256=model_hash, config=cfg, pci=dev.resolve().name,
                          baseline_vram=base,sampled_peak_vram=peak_vram,sampled_peak_rss=peak_ram,
                          peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                          gpu=proof,timing=timings,generated_ids=generated),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runner",type=Path)
    p.add_argument("model",type=Path)
    p.add_argument("--drm",type=Path,required=True)
    p.add_argument("--log",type=Path,required=True)
    bench(p.parse_args())
