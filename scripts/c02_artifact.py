"""Build or load the C02 tokenizer bundle."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

from tokenizers import Tokenizer, __version__
if __package__:
    from .c02_common import roles, sentences, sources, train_bpe
else:
    from c02_common import roles, sentences, sources, train_bpe


def metadata(data):
    digest = sha256(data).hexdigest()
    return {
        "tokenizer_id": f"burda-bpe-v1-sha256-{digest}",
        "vocab_size": 32768,
        "special_tokens": {"PAD": 0, "BOS": 1, "EOS": 2},
        "file": "tokenizer.json",
        "sha256": digest,
        "tokenizers_version": "0.21.4",
    }


def load(directory):
    data = (directory / "tokenizer.json").read_bytes()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest != metadata(data) or __version__ != manifest["tokenizers_version"]:
        raise ValueError("Incompatible manifest, checksum or loader version")
    tokenizer = Tokenizer.from_buffer(data)
    if sorted(tokenizer.get_vocab().values()) != list(range(manifest["vocab_size"])):
        raise ValueError("Invalid vocabulary")
    if [tokenizer.token_to_id(role) for role in roles] != [0, 1, 2]:
        raise ValueError("Invalid special IDs")
    tokenizer.encode_special_tokens = True
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".cache/c02"))
    args = parser.parse_args()
    if args.action == "build":
        if __version__ != "0.21.4":
            parser.error("Install tokenizers==0.21.4")
        model = train_bpe(line for lang in sources for line in sentences(args.cache, lang, "train"))
        args.directory.mkdir(parents=True, exist_ok=True)
        path = args.directory / "tokenizer.json"
        model.save(str(path), pretty=False)
        (args.directory / "manifest.json").write_text(
            json.dumps(metadata(path.read_bytes()), indent=2) + "\n"
        )
    model = load(args.directory)
    sample = "Привет, world!\n  <|pad|><|bos|><|eos|>"
    ids = model.encode(sample, add_special_tokens=False).ids
    if min(ids) < 3 or model.decode(ids) != sample:
        raise ValueError("Round-trip failed")
    print(metadata((args.directory / "tokenizer.json").read_bytes())["tokenizer_id"])
