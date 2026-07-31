import pickle

import pytest
import torch

from donglao_tts.checkpoint import load_checkpoint
from donglao_tts.models.codec import moss_codec


class _UntrustedObject:
    pass


def test_checkpoint_loader_accepts_tensor_primitive_format(tmp_path):
    checkpoint_path = tmp_path / "safe.pt"
    torch.save({"step": 3, "weights": {"value": torch.tensor([1.0])}}, checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    assert checkpoint["step"] == 3
    torch.testing.assert_close(checkpoint["weights"]["value"], torch.tensor([1.0]))


def test_checkpoint_loader_rejects_arbitrary_python_objects(tmp_path):
    checkpoint_path = tmp_path / "unsafe.pt"
    torch.save({"payload": _UntrustedObject()}, checkpoint_path)

    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(checkpoint_path, map_location="cpu")


def test_moss_codec_forwards_pinned_revision(monkeypatch):
    captured = {}

    class _Config:
        sampling_rate = 24_000
        number_channels = 1

    class _Model:
        config = _Config()

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    def fake_from_pretrained(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured.update(kwargs)
        return _Model()

    monkeypatch.setattr(moss_codec.AutoModel, "from_pretrained", fake_from_pretrained)

    moss_codec.MossCodec(
        "owner/codec",
        device="cpu",
        revision="0123456789abcdef",
    )

    assert captured == {
        "repo_id": "owner/codec",
        "revision": "0123456789abcdef",
        "trust_remote_code": True,
        "device": "cpu",
    }
