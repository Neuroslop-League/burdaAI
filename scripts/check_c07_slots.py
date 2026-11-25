"""Exercise the persistent GGML scheduler before exposing it through HTTP."""
import argparse
import json
import resource
import sys
import subprocess
import time
from pathlib import Path

import psutil


def check(runner,model):
    tid=json.loads(Path("artifacts/c02-v1/manifest.json").read_bytes())["tokenizer_id"]
    cmd=[runner,model,"0","1",tid,"1","serve"]
    with subprocess.Popen(cmd,text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL) as proc:
        assert proc.stdout.readline().strip()=="READY"
        peak=0
        def send(line):
            nonlocal peak
            proc.stdin.write(line+"\n");proc.stdin.flush()
            reply=proc.stdout.readline().strip()
            peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
            return reply
        requests=["4 1 568 3546 378 14 5807 3","8 1 3 4 5"]
        def run(active,interrupt=False):
            out={s:[] for s in active}
            begin=time.monotonic()
            for s in active:
                assert send(f"start {s} {requests[s]}")=="OK"
            elapsed={}
            while len(elapsed)<len(active):
                parts=send("tick").split()
                assert parts[0]=="OK"
                for event in parts[1:]:
                    s,id,done=map(int,event.split(":"))
                    if id!=2:out[s].append(id)
                    if done:elapsed[s]=time.monotonic()-begin
                if interrupt and 0 not in elapsed:
                    assert send("cancel 0")=="OK"
                    assert send("start 0 3 1 99999")=="ERR request"
                    elapsed[0]=time.monotonic()-begin
            return out,elapsed
        a,ta=run([0]); b,tb=run([1])
        pair,tp=run([0,1])
        print("single",a,b,"batch",pair,flush=True)
        assert pair==a|b
        interrupted,_=run([0,1],True)
        print("cancel", interrupted,flush=True)
        assert interrupted[1]==b[1]
        assert run([0])[0]==a
        print("seconds_single",ta,tb,"seconds_batch",tp,"batch_tok_s",
              sum(map(len,pair.values()))/max(tp.values()),"peak_rss_bytes",peak,flush=True)
        proc.stdin.write("quit\n");proc.stdin.flush()
        assert proc.wait()==0
        rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        print("kernel_peak_rss_bytes",rss if sys.platform=="darwin" else rss*1024)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runner");p.add_argument("model")
    a=p.parse_args();check(a.runner,a.model)
