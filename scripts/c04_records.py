"""Build the bilingual C04.2 fixture with the released tokenizer."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from c02_artifact import load

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


def split_for(group):
    digest = hashlib.sha256(("c04-v1\n" + group).encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    return "train" if bucket < 9000 else "validation" if bucket < 9500 else "test"


def build(documents, tokenizer, width):
    records = {split: [] for split in SPLITS}
    counts = Counter()
    groups = {split: set() for split in SPLITS}
    for doc in sorted(documents, key=lambda d: d["document_id"]):
        split = split_for(doc["group_id"])
        groups[split].add(doc["group_id"])
        tokens = tokenizer.encode(doc["text"], add_special_tokens=False).ids
        counts[split, doc["language"], "docs"] += 1
        counts[split, doc["language"], "tokens"] += len(tokens)
        ids = [1, *tokens, 2]
        windows = [ids[i:i+width] for i in range(0, len(ids)-1, width-1)]
        assert windows[0] + [t for w in windows[1:] for t in w[1:]] == ids
        for index, window in enumerate(windows):
            assert 2 <= len(window) <= width
            records[split].append(dict(document_id=doc["document_id"], window_index=index, input_ids=window))
    assert sum(map(len, groups.values())) == len(set.union(*groups.values()))
    return records, counts


def main(directory):
    directory.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(s) for s in (ROOT / "fixtures/c04/documents.jsonl").read_text().splitlines()]
    artifact = ROOT / "artifacts/c02-v1"
    tokenizer = load(artifact)
    records, counts = build(docs, tokenizer, 8)
    assert (records, counts) == build(docs[::-1], tokenizer, 8)
    report = {"format_version": 1, "window_size": 8,
              "tokenizer_id": json.loads((artifact / "manifest.json").read_text())["tokenizer_id"],
              "vocab_size": tokenizer.get_vocab_size(), "splits": {}}
    for split, rows in records.items():
        path = directory / f"{split}.jsonl"
        path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode())
        parsed = [json.loads(line) for line in path.read_text().splitlines()]
        assert parsed == rows
        totals = {k: sum(counts[split, lang, k] for lang in ("ru", "en")) for k in ("docs", "tokens")}
        languages = {}
        for lang in ("ru", "en"):
            languages[lang] = {k: counts[split, lang, k] for k in totals}
            languages[lang].update({k+"_pct": round(100*counts[split, lang, k]/v, 2) for k,v in totals.items()})
        report["splits"][split] = dict(languages=languages, windows=len(rows),
                                       stored_ids=sum(len(r["input_ids"]) for r in rows),
                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    # First two windows of different lengths exercise right padding.
    a = records["train"][0]["input_ids"]
    b = next(r["input_ids"] for r in records["train"] if len(r["input_ids"]) != len(a))
    batch = []
    for seq in (a, b):
        pad = max(len(a), len(b)) - len(seq)
        ids = seq + [0]*pad
        mask = [True]*len(seq) + [False]*pad
        batch.append(dict(input_ids=ids, attention_mask=mask,
                          targets=ids[1:]+[-100], loss_mask=mask[1:]+[False]))
    report["batch"] = batch
    (directory / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    for split in SPLITS:
        print(split, report["splits"][split]["languages"])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
