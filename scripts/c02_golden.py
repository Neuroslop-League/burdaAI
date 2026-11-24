"""Verify committed C02.3 reference IDs after loading the release artifact."""
import json
from pathlib import Path

from c02_artifact import load


def verify():
    directory = Path(__file__).resolve().parents[1] / "artifacts/c02-v1"
    golden = json.loads((directory / "golden.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if golden["tokenizer_id"] != manifest["tokenizer_id"]:
        raise ValueError("Reference cases belong to a different tokenizer")
    for _ in range(2):
        tokenizer = load(directory)
        for case in golden["cases"]:
            text, expected = case["text"], case["ids"]
            actual = tokenizer.encode(text).ids
            if actual != expected or any(i < 3 or i >= manifest["vocab_size"] for i in actual):
                raise ValueError(f"{text!r}: expected {expected}, got {actual}")
            framed = [1, *actual, 2, 0]
            for ids, skip, target in [
                (actual, False, text),
                (framed, True, text),
                (framed, False, "<|bos|>" + text + "<|eos|><|pad|>"),
            ]:
                decoded = tokenizer.decode(ids, skip_special_tokens=skip)
                if decoded != target:
                    raise ValueError(f"{ids}, skip={skip}: expected {target!r}, got {decoded!r}")
    print(f"OK: {len(golden['cases'])} cases, 2 independent loads")


if __name__ == "__main__":
    verify()
