import json

import torch

import donglao_tts.api as api
from donglao_tts import DongLaoTTS


class _Info:
    sha = "b" * 40


class _Api:
    def model_info(self, repo_id, revision=None):
        assert repo_id == "DongLao/DongLao-TTS"
        assert revision == "main"
        return _Info()


class _Codec:
    sampling_rate = 24000

    def __init__(self):
        self.saved = None

    def save_audio(self, waveform, path):
        self.saved = (waveform, path)


class _Tokenizer:
    def get_piece_size(self):
        return 1000


def test_from_pretrained_resolves_revision_and_generate(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": {"precision": "float32"}}))
    codec = _Codec()
    ar_model = torch.nn.Linear(2, 2)
    nar_model = torch.nn.Linear(2, 2)
    load_call = {}

    monkeypatch.setattr(api, "HfApi", _Api)
    monkeypatch.setattr(api, "hf_hub_download", lambda **kwargs: str(config_path))

    def fake_load(repo_id, revision, device):
        load_call.update(repo_id=repo_id, revision=revision, device=device)
        return ar_model, nar_model, codec, _Tokenizer(), object(), 1024, 8

    monkeypatch.setattr(api, "load_from_hub", fake_load)
    monkeypatch.setattr(api, "Pipeline", lambda: "pipeline")

    tts = DongLaoTTS.from_pretrained(revision="main", device="cpu")

    assert tts.revision == "b" * 40
    assert tts.sample_rate == 24000
    assert load_call["revision"] == "b" * 40
    assert load_call["device"] == torch.device("cpu")

    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")
    output = tmp_path / "result.wav"
    generated = torch.tensor([[0.25, -0.5]])
    generation_call = {}

    def fake_generate(config, *args, **kwargs):
        generation_call.update(config=config, args=args, kwargs=kwargs)
        return generated, torch.zeros(1, 2)

    monkeypatch.setattr(api, "generate_sample", fake_generate)
    result = tts.generate(
        "Xin chào",
        reference_audio=reference,
        reference_text="Hello",
        output_path=output,
        max_frames=12,
        temperature=0.8,
        top_k=20,
    )

    assert result is generated
    assert generation_call["config"]["sample"]["target_text"] == "Xin chào"
    assert generation_call["config"]["sample"]["temperature"] == 0.8
    assert generation_call["kwargs"]["max_frames"] == 12
    assert codec.saved == (generated, str(output.resolve()))


def test_generate_validates_user_input(tmp_path):
    tts = DongLaoTTS(
        repo_id="repo",
        revision="sha",
        device="cpu",
        dtype=torch.float32,
        ar_model=object(),
        nar_model=object(),
        codec=_Codec(),
        tokenizer=_Tokenizer(),
        special_tokens=object(),
        codebook_size=1024,
        num_quantizers=8,
        pipeline=object(),
    )

    missing = tmp_path / "missing.wav"
    try:
        tts.generate("text", reference_audio=missing, reference_text="transcript")
    except FileNotFoundError as exc:
        assert "reference audio does not exist" in str(exc)
    else:
        raise AssertionError("expected a missing reference file to fail")


def test_generate_uses_sampling_defaults(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")
    codec = _Codec()
    tts = DongLaoTTS(
        repo_id="repo",
        revision="sha",
        device="cpu",
        dtype=torch.float32,
        ar_model=object(),
        nar_model=object(),
        codec=codec,
        tokenizer=_Tokenizer(),
        special_tokens=object(),
        codebook_size=1024,
        num_quantizers=8,
        pipeline=object(),
    )
    generation_call = {}

    def fake_generate(config, *args, **kwargs):
        generation_call.update(config)
        return torch.ones(1, 1), torch.ones(1, 1)

    monkeypatch.setattr(api, "generate_sample", fake_generate)
    tts.generate("text", reference_audio=reference, reference_text="transcript")

    assert generation_call["sample"]["temperature"] == 0.8
    assert generation_call["sample"]["top_k"] == 10
    assert generation_call["sample"]["sentence_pause_ms"] == 180
    assert generation_call["sample"]["leading_silence_ms"] == 20
    assert generation_call["sample"]["trailing_silence_ms"] == 20


def test_generate_batch_reuses_shared_reference(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")
    tts = DongLaoTTS(
        repo_id="repo",
        revision="sha",
        device="cpu",
        dtype=torch.float32,
        ar_model=object(),
        nar_model=object(),
        codec=_Codec(),
        tokenizer=_Tokenizer(),
        special_tokens=object(),
        codebook_size=1024,
        num_quantizers=8,
        pipeline=object(),
    )
    call = {}
    expected = [torch.tensor([[1.0]]), torch.tensor([[2.0]])]

    def fake_generate_batch(config, texts, *args, **kwargs):
        call.update(config=config, texts=texts, args=args, kwargs=kwargs)
        return expected, torch.zeros(1, 1)

    monkeypatch.setattr(api, "generate_batch_samples", fake_generate_batch)
    result = tts.generate_batch(
        ["Câu một.", "Câu hai."],
        reference_audio=reference,
        reference_text="reference",
    )

    assert result is expected
    assert call["texts"] == ["Câu một.", "Câu hai."]
    assert call["config"]["sample"]["temperature"] == 0.8
    assert call["config"]["sample"]["top_k"] == 10
    assert call["config"]["sample"]["sentence_pause_ms"] == 180
    assert call["config"]["sample"]["leading_silence_ms"] == 20
    assert call["config"]["sample"]["trailing_silence_ms"] == 20


def test_generate_stream_yields_codec_chunks(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")
    tts = DongLaoTTS(
        repo_id="repo",
        revision="sha",
        device="cpu",
        dtype=torch.float32,
        ar_model=object(),
        nar_model=object(),
        codec=_Codec(),
        tokenizer=_Tokenizer(),
        special_tokens=object(),
        codebook_size=1024,
        num_quantizers=8,
        pipeline=object(),
    )
    decoded_chunks = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0, 5.0]])]
    call = {}

    def fake_stream(config, *args, **kwargs):
        call.update(config=config, args=args, kwargs=kwargs)
        yield from decoded_chunks

    monkeypatch.setattr(api, "generate_sample_stream", fake_stream)
    result = list(tts.generate_stream(
        "Câu một. Câu hai.",
        reference_audio=reference,
        reference_text="reference",
        chunk_frames=5,
    ))

    assert len(result) == 2
    assert torch.equal(result[0], decoded_chunks[0])
    assert torch.equal(result[1], decoded_chunks[1])
    assert call["config"]["sample"]["target_text"] == "Câu một. Câu hai."
    assert call["config"]["sample"]["sentence_pause_ms"] == 180
    assert call["config"]["sample"]["leading_silence_ms"] == 20
    assert call["config"]["sample"]["trailing_silence_ms"] == 20
    assert call["kwargs"]["max_frames"] == 200
    assert call["kwargs"]["chunk_frames"] == 5


def test_generate_batch_validates_inputs(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"wav")
    tts = DongLaoTTS(
        repo_id="repo",
        revision="sha",
        device="cpu",
        dtype=torch.float32,
        ar_model=object(),
        nar_model=object(),
        codec=_Codec(),
        tokenizer=_Tokenizer(),
        special_tokens=object(),
        codebook_size=1024,
        num_quantizers=8,
        pipeline=object(),
    )

    try:
        tts.generate_batch(
            ["one", "two"],
            reference_audio=reference,
            reference_text="reference",
            output_paths=[tmp_path / "one.wav"],
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected mismatched output_paths to fail")
