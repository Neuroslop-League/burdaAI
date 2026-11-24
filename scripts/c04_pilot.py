"""Build C04.3 shards from the versioned, reviewed Wikipedia excerpts."""
import argparse
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from c02_common import sentences
from c02_artifact import load
from c04_records import ROOT, SPLITS, split_for

SOURCE = ROOT / "data/c04-pilot"


def normalize(s):
    return " ".join(unicodedata.normalize("NFC", s).casefold().split())


def shingles(words):
    return set(zip(*(words[i:] for i in range(5))))


def near(a, b):
    common = len(a & b)
    return common >= 20 and (common >= .85*len(a | b) or common >= .9*min(len(a), len(b)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(out, cache, reverse):
    sources = (SOURCE / "approved.jsonl").read_bytes()
    docs = [json.loads(s) for s in sources.splitlines()]
    if reverse:
        docs.reverse()
    tokdir = ROOT / "artifacts/c02-v1"
    tokenizer = load(tokdir)
    manifest = json.loads((tokdir / "manifest.json").read_text())
    ud = {}
    for part in ("train", "dev"):
        rows = [normalize(s) for lang in ("ru", "en") for s in sentences(cache, lang, part)]
        ud[part] = [(s, shingles(re.findall(r"[^\W\d_]+", s))) for s in rows if len(s.split()) >= 10]
    keys = set()
    signatures = []
    topics = {}
    partitions = {k: [] for k in SPLITS}
    for doc in sorted(docs, key=lambda d: d["document_id"]):
        text, key = doc["text"], doc["document_id"]
        lang = doc["language"]
        assert "\ufffd" not in text and all(unicodedata.category(c) != "Cc" or c in "\r\n\t" for c in text)
        words = re.findall(r"[^\W\d_]+", text, re.UNICODE)
        letters = "".join(words)
        script = "CYRILLIC" if lang == "ru" else "LATIN"
        assert (doc["status"] == "approved" and len(words) >= 50 and key not in keys
                and doc["text_sha256"] == sha(text.encode())
                and sum(script in unicodedata.name(c, "") for c in letters) >= .6*len(letters))
        keys.add(key)
        norm = normalize(text)
        sig = shingles(re.findall(r"[^\W\d_]+", norm))
        # Registry contains selected survivors. Fail if a duplicate escaped review.
        assert all(norm != other and not near(sig, sh) for other, sh in signatures)
        signatures.append((norm, sig))
        split = split_for(key)
        for part in ("train", "dev"):
            if any(s in norm or near(sig, sh) for s, sh in ud[part]):
                if part == "train":
                    split = "train"
                    break
                if split == "test":
                    split = "validation"
        topic = doc["wikidata"]
        assert topic not in topics or topics[topic] == split
        topics[topic] = split
        partitions[split].append((key, lang, tokenizer.encode(text).ids))
    out.mkdir(parents=True, exist_ok=True)
    report = dict(format_version=1, source_revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                  source_sha256=sha(sources), tokenizer_id=manifest["tokenizer_id"],
                  vocab_size=manifest["vocab_size"], window_size=4096, splits={})
    for split, docs_in_split in partitions.items():
        counts = Counter()
        records = []
        for key, lang, tokens in docs_in_split:
            counts[lang, "documents"] += 1
            counts[lang, "tokens"] += len(tokens)
            ids = [1, *tokens, 2]
            for index, i in enumerate(range(0, len(ids)-1, 4095)):
                window = ids[i:i+4096]
                assert 2 <= len(window) <= 4096 and all(0 <= t < manifest["vocab_size"] for t in window)
                records.append(dict(document_id=key, window_index=index, input_ids=window))
        data = "".join(json.dumps(r, separators=(",", ":"))+"\n" for r in records).encode()
        name = split+"-00000.jsonl"
        (out/name).write_bytes(data)
        stats = {}
        for lang in ("ru", "en"):
            stats[lang] = {k: counts[lang,k] for k in ("documents", "tokens")}
            for k in ("documents", "tokens"):
                stats[lang][k+"_pct"] = round(100*counts[lang,k]/sum(counts[l,k] for l in ("ru","en")), 2)
        report["splits"][split] = dict(file=name, sha256=sha(data), bytes=len(data), windows=len(records),
                                       stored_ids=sum(len(r["input_ids"]) for r in records), languages=stats)
        assert min(counts[l,"documents"] for l in ("ru","en")) >= (1 if split == "train" else 8)
        if split == "train":
            assert counts["ru","tokens"] > counts["en","tokens"]
        print(split, stats)
    for a,b in itertools.combinations(partitions.values(),2):
        assert {row[0] for row in a}.isdisjoint(row[0] for row in b)
    (out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    build(args.out, args.cache, args.reverse)
