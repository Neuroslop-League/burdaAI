"""Convert a verified C07.1 reference bundle to block-quantized GGUF."""
import argparse
import json
import os
import tempfile
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

from burdaai.decoder import Config
from scripts.c07_export import digest
from scripts.check_c07_bundle import layout


def convert(src, output, kind):
    manifest = json.loads((src/"manifest.json").read_bytes())
    names = {"config.json", "tokenizer.json", "model.safetensors", "tokenizer-manifest.json"}
    if (manifest["schema_version"], manifest["profile"], set(manifest["files"])) != (1,"reference-f32",names):
        raise ValueError("Expected reference-f32 v1")
    for name, info in manifest["files"].items():
        if (src/name).stat().st_size != info["bytes"] or digest(src/name) != info["sha256"]:
            raise ValueError(name)
    c = Config(**json.loads((src/"config.json").read_bytes()))
    tok = json.loads((src/"tokenizer.json").read_bytes())
    if c.tokenizer_id != manifest["tokenizer_id"] or c.tokenizer_id != "burda-bpe-v1-sha256-"+manifest["files"]["tokenizer.json"]["sha256"]:
        raise ValueError("Tokenizer mismatch")
    vocab = sorted(tok["model"]["vocab"].items(), key=lambda item:item[1])
    if [i for _,i in vocab] != list(range(c.vocab_size)):
        raise ValueError("Vocabulary mismatch")
    if output.exists():
        raise FileExistsError(output)
    qtype = gguf.GGMLQuantizationType[kind]
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        path = Path(tmp)/"model.gguf"
        w = gguf.GGUFWriter(path,"llama",use_temp_file=True)
        w.add_name("burdaAI")
        w.add_string("burda.tokenizer_id",c.tokenizer_id)
        w.add_string("burda.source_sha256",manifest["files"]["model.safetensors"]["sha256"])
        w.add_uint32("general.quantization_version",2)
        w.add_file_type(gguf.LlamaFileType.ALL_F32 if kind=="F32" else getattr(gguf.LlamaFileType,"MOSTLY_"+kind))
        w.add_context_length(4096)
        w.add_embedding_length(2048)
        w.add_block_count(20)
        w.add_feed_forward_length(5632)
        w.add_head_count(16)
        w.add_head_count_kv(4)
        w.add_rope_freq_base(10000)
        w.add_layer_norm_rms_eps(1e-5)
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("gpt-2")
        w.add_token_list([s for s,_ in vocab])
        w.add_token_types([3]*3+[1]*(len(vocab)-3))
        w.add_token_merges([" ".join(pair) for pair in tok["model"]["merges"]])
        w.add_bos_token_id(1)
        w.add_eos_token_id(2)
        w.add_pad_token_id(0)
        mapping = {"embedding.weight":"token_embd.weight","norm.weight":"output_norm.weight"}
        for i in range(20):
            for a,b in zip(("attn_norm","attn.q","attn.k","attn.v","attn.o","ffn_norm","gate","up","down"),
                           ("attn_norm","attn_q","attn_k","attn_v","attn_output","ffn_norm","ffn_gate","ffn_up","ffn_down")):
                mapping[f"blocks.{i}.{a}.weight"] = f"blk.{i}.{b}.weight"
        squared = count = 0
        shapes = layout(c)
        with safe_open(src/"model.safetensors",framework="numpy") as tensors:
            if set(mapping) != set(tensors.keys()):
                raise ValueError("Unexpected tensor inventory")
            for key, name in mapping.items():
                a = tensors.get_tensor(key)
                if a.dtype != np.float32 or a.shape != shapes[key]:
                    raise ValueError(key)
                if a.ndim == 1 or kind=="F32":
                    w.add_tensor(name,a)
                    continue
                chunks = []
                for i in range(0,len(a),128):
                    block = a[i:i+128]
                    packed = gguf.quantize(block,qtype)
                    error = gguf.dequantize(packed,qtype)-block
                    squared += np.sum(error*error,dtype=np.float64)
                    count += error.size
                    chunks.append(packed)
                w.add_tensor(name,np.concatenate(chunks),raw_dtype=qtype)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        os.rename(path,output)
    print(kind,output.stat().st_size,"bytes, matrix RMSE",float(np.sqrt(squared/count)) if count else 0.0,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("kind",choices=["Q4_0","Q4_1","F32"])
    args=p.parse_args()
    convert(args.bundle,args.output,args.kind)
