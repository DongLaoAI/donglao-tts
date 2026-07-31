import torch

from donglao_tts.cli import smoke_test_hub


class _Info:
    sha = "a" * 40


class _Api:
    def model_info(self, repo_id, revision=None):
        assert repo_id == "DongLao/DongLao-TTS"
        assert revision is None
        return _Info()


class _Tokenizer:
    def get_piece_size(self):
        return 1000


class _Codec:
    pass


class _TTS:
    ar_model = torch.nn.Linear(2, 3)
    nar_model = torch.nn.Linear(3, 4)
    codec = _Codec()
    tokenizer = _Tokenizer()

    @classmethod
    def from_pretrained(cls, repo_id, revision, device):
        assert repo_id == "DongLao/DongLao-TTS"
        assert revision == "a" * 40
        assert device == torch.device("cpu")
        return cls()


def test_load_only_smoke_test(monkeypatch, capsys):
    monkeypatch.setattr(smoke_test_hub, "HfApi", _Api)
    monkeypatch.setattr(
        smoke_test_hub,
        "_validate_bundle",
        lambda repo_id, revision: (
            {"codec": {"bundled": True}},
            {"model": {"precision": "float32"}},
        ),
    )
    monkeypatch.setattr(smoke_test_hub, "DongLaoTTS", _TTS)

    smoke_test_hub.main(["--device", "cpu"])

    output = capsys.readouterr().out
    assert "DongLao/DongLao-TTS@" + "a" * 40 in output
    assert "AR=9 params" in output
    assert "NAR=16 params" in output
    assert "PASS: installed package loaded all native weights" in output


def test_inference_arguments_must_be_complete():
    try:
        smoke_test_hub.main(["--ref-audio", "reference.wav"])
    except SystemExit as exc:
        assert "must be supplied together" in str(exc)
    else:
        raise AssertionError("expected incomplete inference arguments to fail")
