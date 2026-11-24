"""CPU checkpoint interchange between the reference decoder and C05."""
from dataclasses import asdict

import torch

from .decoder import Config, Decoder


def save_decoder(path, model):
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_decoder(path, expected: Config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if Config(**checkpoint["config"]) != expected:
        raise ValueError("Checkpoint architecture or tokenizer differs from expected config")
    with torch.device("meta"):
        model = Decoder(expected)
    model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
    return model
