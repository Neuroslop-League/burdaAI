"""Run from the repository root with tokenizers==0.21.4."""
import sys
from pathlib import Path
from tokenizers import Tokenizer
from c02_common import fetch, roles, sentences, sources, train_bpe

cache = Path(sys.argv[1])
train = [line for lang in sources for line in sentences(cache, lang, "train")]
seen = set(train)
dev = {}
for lang in sources:
    raw = sentences(cache, lang, "dev")
    dev[lang] = [line for line in raw if line not in seen]
    print(lang, "excluded", len(raw) - len(dev[lang]), "/", len(raw))
dev["mixed"] = [line for pair in zip(dev["ru"], dev["en"]) for line in pair]

custom = train_bpe(train)
gpt = Tokenizer.from_file(str(fetch(cache, "gpt2.json",
    "https://huggingface.co/openai-community/gpt2/resolve/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json")))
gpt.add_special_tokens(roles[:2])
for label, tokenizer, special in [
    ("custom", custom, roles), ("gpt2", gpt, roles[:2] + ["<|endoftext|>"]),
]:
    tokenizer.encode_special_tokens = True  # Treat literal spellings as ordinary input.
    reserved = {tokenizer.token_to_id(s) for s in special}
    def count(line):
        ids = tokenizer.encode(line, add_special_tokens=False).ids
        assert reserved.isdisjoint(ids)
        assert tokenizer.decode(ids, skip_special_tokens=False) == line
        return len(ids)

    for line in ["", " е\u0308  Ё\n🙂𐍈\t", *roles, "<|endoftext|>"]:
        count(line)
    print(label, "vocab", tokenizer.get_vocab_size())
    for lang, lines in dev.items():
        blocks = ["\n".join(lines[i:i + 256]) for i in range(0, len(lines), 256)]
        for unit, texts in [("sentence", lines), ("block", blocks)]:
            lengths = [count(line) for line in texts]
            size = sum(len(line.encode()) for line in texts)
            print(lang, unit, len(texts), size, sum(lengths), round(sum(lengths)/size, 5),
                  sum(n + 2 > 4096 for n in lengths))
