"""Run python -m scripts.check_c03_handoff from the repository root."""
import gc
import tempfile
from pathlib import Path
from time import monotonic

import torch

from burdaai.checkpoint import load_decoder, save_decoder
from burdaai.decoder import Config, Decoder


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    config = Config.load(Path(__file__).resolve().parents[1] / "configs/c03-fixture.json")
    model = Decoder(config).eval()
    short = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    batch = torch.tensor([[1, 2, 3, 0, 0], [3, 4, 5, 6, 7]])
    mask = torch.tensor([[True]*3 + [False]*2, [True]*5])
    single = model(short)
    paired = model(batch, mask)
    error = (single[0] - paired[0, :3]).abs().max().item()
    assert error <= 1e-4 and paired.isfinite().all() and paired[~mask].count_nonzero() == 0
    print(f"padded batch: max absolute error {error:.9g}, padded logits exactly zero", flush=True)

    for length in (1, 4097, 4096):
        ids = torch.arange(length).remainder(config.vocab_size)[None]
        start = monotonic()
        if length == 4097:
            try:
                model(ids)
            except ValueError as exc:
                if str(exc) != "Invalid batch or sequence length":
                    raise
                print(f"T={length}: rejected", flush=True)
                continue
            raise AssertionError("4097 accepted")
        logits = model(ids)
        assert logits.shape == (1, length, config.vocab_size) and logits.isfinite().all()
        print(f"T={length}: {tuple(logits.shape)}, finite, {monotonic()-start:.2f}s", flush=True)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "decoder.pt"
        save_decoder(path, model)
        del model
        gc.collect()
        restored = load_decoder(path, config).eval()
        result = restored(batch, mask)
        torch.testing.assert_close(result, paired, rtol=0, atol=0)
        print(f"reload: {path.stat().st_size} bytes, max absolute error 0", flush=True)


if __name__ == "__main__":
    main()
