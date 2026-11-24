"""Compare a GGUF backend to the four accepted C07.5 Rust FP32 traces."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def compare(work, runner, models):
    directory = work/"bundle"
    config = json.loads((directory/"config.json").read_bytes())
    tok = Tokenizer.from_file(str(directory/"tokenizer.json"))
    tok.encode_special_tokens = True
    prompt = [1,*tok.encode("Привет, мир!",add_special_tokens=False).ids]
    steps = json.loads((work/"rust-logits/steps.json").read_bytes())
    for model in models:
        ids = prompt.copy()
        sq = peak = n = matches = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"logits.f32"
            for i,step in enumerate(steps):
                run = subprocess.run([runner,model,"0","1",config["tokenizer_id"],",".join(map(str,ids)),
                                      "generate",path],capture_output=True,text=True)
                if run.returncode:
                    raise RuntimeError(run.stderr[-2000:])
                ref = np.fromfile(work/f"rust-logits/{i}.f32",dtype="<f4").reshape(step["shape"])[-1]
                other = np.fromfile(path,dtype="<f4")
                delta = other.astype(np.float64)-ref
                peak = max(peak,float(abs(delta).max()))
                sq += delta @ delta
                n += len(delta)
                matches += int(other.argmax()==step["id"])
                ids.append(step["id"])
        print(model.name,"bytes",model.stat().st_size,"logit RMSE",float(np.sqrt(sq/n)),
              "max_abs",peak,"greedy",matches,"/",len(steps),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("runner",type=Path)
    p.add_argument("models",type=Path,nargs="+")
    args=p.parse_args()
    compare(args.work,args.runner,args.models)
