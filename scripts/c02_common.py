from hashlib import sha256
from urllib.request import urlretrieve

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sources = {
    "ru": ("Russian-GSD", "47c45deefdcb3ac15e2f33e3093c8d10c1b6b36e", "ru_gsd"),
    "en": ("English-EWT", "4dc8e10cf32352e11ab2c46e024b19853b91546e", "en_ewt"),
}


def fetch(cache, name, url):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists():
        temporary = path.with_suffix(".part")
        urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def sentences(cache, language, split):
    repo, revision, prefix = sources[language]
    name = f"{prefix}-ud-{split}.conllu"
    url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_{repo}/{revision}/{name}"
    path = fetch(cache, name, url)
    if split == "train" and sha256(path.read_bytes()).hexdigest() != TRAIN_HASH[language]:
        raise ValueError(f"Checksum mismatch: {path}")
    with path.open(encoding="utf-8") as source:
        return [line[9:].removesuffix("\n") for line in source if line.startswith("# text = ")]


TRAIN_HASH = {
    "ru": "2e6aeb49f69455fdfa9ed948e21a3fb3783a653d5a6439268dc2cbcfcab70353",
    "en": "a049fc40822e21593ff89ccd0a4b081443f09fae71b61f813c624aa598f6c566",
}
roles = ["<|pad|>", "<|bos|>", "<|eos|>"]


def train_bpe(train):
    custom = Tokenizer(models.BPE())
    custom.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    custom.decoder = decoders.ByteLevel()
    custom.train_from_iterator(train, trainers.BpeTrainer(
        vocab_size=32768, special_tokens=roles, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    ))
    assert sorted(custom.get_vocab().values()) == list(range(32768))
    assert [custom.token_to_id(role) for role in roles] == [0, 1, 2]
    return custom
