"""Mac CLI benchmark, single request. Use fresh output directory."""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil


def run(args, length, count, threads, cache, trace=False):
    name=f"{length}-{count}-t{threads}-{cache}"
    directory=args.output/name
    directory.mkdir()
    command=[args.cli,"--gguf",args.gguf,"--runner",args.runner,"--tokenizer","artifacts/c02-v1",
             "--device","cpu","--placement","cpu","--text"," a"*(length-1),"--max-new-tokens",
             str(count),"--kv",cache,"--threads",str(threads)]
    if (length,count)==(256,64):
        command.append("--benchmark-256x64")
    env=os.environ.copy()
    if trace:
        env["BURDA_TRACE_ALL"]=str(directory)
    start=time.monotonic()
    peak=0
    with (directory/"log").open("w") as log,(directory/"text").open("w") as out:
        with subprocess.Popen(command,stdout=out,stderr=log,env=env) as p:
            proc=psutil.Process(p.pid)
            while p.poll() is None:
                try:
                    peak=max(peak,sum(child.memory_info().rss for child in [proc,*proc.children(recursive=True)]))
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.02)
            if p.returncode:
                raise RuntimeError(directory/"log")
    wall=time.monotonic()-start
    log=(directory/"log").read_text()
    fields=dict(s.split("=") for s in re.search(r"^prompt=.*",log,re.M)[0].split())
    assert int(fields["prompt"])==length and int(fields["emitted"])==count
    kv_mib=float(re.search(r"KV self size\s*=\s*([\d.]+) MiB",log)[1])
    result=dict(name=name,wall_s=round(wall,3),rss_mib=round(peak/2**20,2),kv_mib=kv_mib,**fields)
    print(json.dumps(result),flush=True)
    return directory


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("cli","gguf","runner","output"):
        p.add_argument(key,type=Path)
    args=p.parse_args()
    args.output.mkdir()
    a,b=[run(args,7,4,4,mode,True) for mode in ("off","on")]
    for i in range(4):
        x,y=[np.fromfile(d/f"{i}.f32",dtype="<f4") for d in (a,b)]
        error=float(abs(x-y).max())
        print(i,error,int(x.argmax()),int(y.argmax()),flush=True)
        assert x.size==y.size==32768 and error<=.15 and x.argmax()==y.argmax()
    for length,count,threads in [(256,64,4),(32,4,2),(32,4,4),(128,4,2),(128,4,4)]:
        for mode in ("on","off"):
            run(args,length,count,threads,mode)

    assert (args.output/"256-64-t4-on/text").read_bytes() == (args.output/"256-64-t4-off/text").read_bytes()
