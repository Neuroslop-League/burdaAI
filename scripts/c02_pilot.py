"""Run from the repository root with tokenizers==0.21.4."""
import sys
from pathlib import Path
from urllib.request import urlretrieve

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

cache = Path(sys.argv[1])
cache.mkdir(parents=True, exist_ok=True)
sources = {
    "ru": ("Russian-GSD", "47c45deefdcb3ac15e2f33e3093c8d10c1b6b36e", "ru_gsd"),
    "en": ("English-EWT", "4dc8e10cf32352e11ab2c46e024b19853b91546e", "en_ewt"),
}


def fetch(name, url):
    path = cache / name
    if not path.exists():
        temporary = path.with_suffix(".part")
        urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def sentences(language, split):
    repo, revision, prefix = sources[language]
    name = f"{prefix}-ud-{split}.conllu"
    url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_{repo}/{revision}/{name}"
    with fetch(name, url).open(encoding="utf-8") as source:
        return [line[9:].removesuffix("\n") for line in source if line.startswith("# text = ")]


train = [line for lang in sources for line in sentences(lang, "train")]
seen = set(train)
dev = {}
for lang in sources:
    raw = sentences(lang, "dev")
    dev[lang] = [line for line in raw if line not in seen]
    print(lang, "excluded", len(raw) - len(dev[lang]), "/", len(raw))
dev["mixed"] = [line for pair in zip(dev["ru"], dev["en"]) for line in pair]

roles = ["<|pad|>", "<|bos|>", "<|eos|>"]
custom = Tokenizer(models.BPE())
custom.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
custom.decoder = decoders.ByteLevel()
custom.train_from_iterator(train, trainers.BpeTrainer(
    vocab_size=32768, special_tokens=roles, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
))
assert sorted(custom.get_vocab().values()) == list(range(32768))
assert [custom.token_to_id(role) for role in roles] == [0, 1, 2]

gpt = Tokenizer.from_file(str(fetch("gpt2.json",
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
