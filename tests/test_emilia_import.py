import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch


RAW_TO_COMPILED = Path(__file__).resolve().parents[1] / "scripts" / "raw_to_compiled"
sys.path.insert(0, str(RAW_TO_COMPILED))
import convert_emilia_en as emilia  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_sentencepiece(monkeypatch):
    monkeypatch.setattr(
        emilia.spm,
        "SentencePieceProcessor",
        lambda **kwargs: SimpleNamespace(encode=lambda text, out_type: [1]),
    )


def _audio_bytes():
    destination = io.BytesIO()
    sf.write(destination, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return destination.getvalue()


def _webdataset_tar(metadata):
    destination = io.BytesIO()
    with tarfile.open(fileobj=destination, mode="w") as archive:
        payloads = {
            f"{metadata['id']}.json": json.dumps(metadata).encode(),
            f"{metadata['id']}.mp3": _audio_bytes(),
        }
        for name, payload in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    destination.seek(0)
    return destination.getvalue()


class _FakeFilesystem:
    def __init__(self, payload):
        self.payload = payload
        self.opened = []

    def open(self, path, mode):
        assert mode == "rb"
        self.opened.append(path)
        return io.BytesIO(self.payload)


class _FakeCodec:
    num_channels = 1
    sampling_rate = 16000

    def encode(self, waveform):
        assert waveform.shape == (1, 1600)
        return torch.zeros((8, 3), dtype=torch.long)


class _FakePipeline:
    def phonemize_batch(self, texts):
        assert texts == ["Hello world"]
        return ["həlˈoʊ wˈɜːld"]


class _EmptyPipeline:
    def phonemize_batch(self, texts):
        assert texts == ["Hello world"]
        return [""]


def test_iter_webdataset_records_pairs_json_and_audio():
    metadata = {
        "id": "EN_B00000_S00000_W000000",
        "text": "Hello world",
        "speaker": "EN_B00000_S00000",
        "language": "en",
    }
    records = list(emilia.iter_webdataset_records(io.BytesIO(_webdataset_tar(metadata))))

    assert len(records) == 1
    assert records[0][0] == metadata
    assert records[0][1].startswith(b"RIFF")


def test_convert_emilia_en_streams_and_commits(tmp_path, monkeypatch):
    metadata = {
        "id": "EN_B00000_S00000_W000000",
        "text": "Hello world",
        "speaker": "EN_B00000_S00000",
        "language": "en",
    }
    filesystem = _FakeFilesystem(_webdataset_tar(metadata))
    repo_info = SimpleNamespace(
        sha=emilia.DEFAULT_REVISION,
        siblings=[SimpleNamespace(rfilename="Emilia/EN/EN-B000000.tar")],
    )
    config = tmp_path / "config.yaml"
    tokenizer = tmp_path / "spm.model"
    tokenizer.write_bytes(b"test tokenizer")
    config.write_text(
        "codec:\n"
        "  num_quantizers: 8\n"
        "tokenizer:\n"
        f"  model_path: {tokenizer}\n",
        encoding="utf-8",
    )
    compiled_calls = []

    def fake_compile(manifests, output, tokenizer_path, **kwargs):
        compiled_calls.append((manifests, output, tokenizer_path, kwargs))
        return {"shards": [{"name": "shard-00000"}]}

    monkeypatch.setattr(emilia, "compile_dataset", fake_compile)
    state = emilia.convert_emilia_en(
        config_path=config,
        output_path=tmp_path / "compiled",
        work_dir=tmp_path / "work",
        repo_info=repo_info,
        filesystem=filesystem,
        codec=_FakeCodec(),
        pipeline=_FakePipeline(),
        resume=True,
        strict=True,
    )

    assert state["compiled"] == 1
    assert state["seen"] == 1
    assert state["next_shard"] == 1
    assert state["next_sample"] == 0
    assert state["skipped"] == 0
    assert len(compiled_calls) == 1
    assert compiled_calls[0][0][0][1:] == (emilia.DEFAULT_CORPUS, "en")
    assert filesystem.opened[0].endswith("/Emilia/EN/EN-B000000.tar")
    assert (tmp_path / "work" / "pending.phon.jsonl").read_text() == ""


def test_convert_emilia_en_skips_empty_phoneme(tmp_path, monkeypatch):
    metadata = {
        "id": "EN_B00000_S00000_W000000",
        "text": "Hello world",
        "speaker": "EN_B00000_S00000",
        "language": "en",
    }
    config = tmp_path / "config.yaml"
    tokenizer = tmp_path / "spm.model"
    tokenizer.write_bytes(b"test tokenizer")
    config.write_text(
        "codec:\n"
        "  num_quantizers: 8\n"
        "tokenizer:\n"
        f"  model_path: {tokenizer}\n",
        encoding="utf-8",
    )
    repo_info = SimpleNamespace(
        sha=emilia.DEFAULT_REVISION,
        siblings=[SimpleNamespace(rfilename="Emilia/EN/EN-B000000.tar")],
    )
    monkeypatch.setattr(
        emilia,
        "compile_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid sample must not reach compile_dataset")
        ),
    )

    state = emilia.convert_emilia_en(
        config_path=config,
        output_path=tmp_path / "compiled",
        work_dir=tmp_path / "work",
        repo_info=repo_info,
        filesystem=_FakeFilesystem(_webdataset_tar(metadata)),
        codec=_FakeCodec(),
        pipeline=_EmptyPipeline(),
        strict=False,
    )

    assert state["compiled"] == 0
    assert state["skipped"] == 1
    assert state["seen"] == 1
    assert (tmp_path / "work" / "pending.phon.jsonl").read_text() == ""
