"""Exercise the local server with the real GGUF and compare with the CLI."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request


def check(a):
    directory=Path("artifacts/c02-v1")
    cmd=[a.server,a.runner,a.model,directory,str(a.port)]
    base=f"http://127.0.0.1:{a.port}"
    def request(payload):
        req=Request(base+"/generate",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
        try:
            with urlopen(req,timeout=90) as response:return response.status,json.load(response)
        except HTTPError as e:return e.code,json.load(e)
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        data=(directory/"tokenizer.json").read_bytes()+b" "
        (d/"tokenizer.json").write_bytes(data)
        meta=json.loads((directory/"manifest.json").read_bytes())
        meta["sha256"]=hashlib.sha256(data).hexdigest()
        meta["tokenizer_id"]="burda-bpe-v1-sha256-"+meta["sha256"]
        (d/"manifest.json").write_text(json.dumps(meta))
        bad=subprocess.run([a.server,a.runner,a.model,d,str(a.port)],capture_output=True,timeout=30)
        assert bad.returncode and b"incompatible tokenizer" in bad.stderr
        with (d/"server.log").open("w") as log:
            server=subprocess.Popen(cmd,stderr=log)
            try:
                for _ in range(200):
                    if server.poll() is not None:raise RuntimeError((d/"server.log").read_text()[-1000:])
                    try:
                        with urlopen(base+"/ready",timeout=1) as r:assert json.load(r)=={"ready":True}
                        break
                    except URLError:time.sleep(.05)
                else:raise RuntimeError("server readiness timeout")
                q={"text":"Привет, мир!","max_new_tokens":4}
                status,result=request(q)
                assert status==200 and len(result["ids"])==4
                cli=subprocess.check_output([a.cli,"--gguf",a.model,"--runner",a.runner,"--tokenizer",directory,
                                             "--text",q["text"],"--max-new-tokens","4"],stderr=subprocess.DEVNULL).decode()
                assert cli==result["text"]+"\n"
                assert request({"text":" a"*4095,"max_new_tokens":1})[0]==422
                assert request({"bad":1})[0]==400
                with ThreadPoolExecutor(10) as ex:
                    futures=[ex.submit(request,{"text":" a"*32,"max_new_tokens":32}) for _ in range(10)]
                    statuses=[f.result()[0] for f in futures]
                    assert 429 in statuses and set(statuses)=={200,429}
                    pending=ex.submit(request,{"text":" a"*32,"max_new_tokens":32})
                    time.sleep(.15)
                    server.send_signal(signal.SIGINT)
                    assert pending.result()[0]==200
                assert server.wait(timeout=15)==0
                print("ready 200, generate 200, invalid 400, overflow 422, queue",statuses,
                      "incompatible model rejected, shutdown 0; IDs",result["ids"])
            finally:
                if server.poll() is None:
                    server.kill();server.wait()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("server","cli","runner","model"):p.add_argument(name,type=Path)
    p.add_argument("--port",type=int,default=18973)
    check(p.parse_args())
