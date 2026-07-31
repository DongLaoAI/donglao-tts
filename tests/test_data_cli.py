import json
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from donglao_tts.cli._io import atomic_text_writer
from donglao_tts.cli.build_phoneme_corpus import build_phoneme_corpus
from donglao_tts.cli.init_config import initialize_config
from donglao_tts.cli.phonemize_manifest import phonemize_manifest
from donglao_tts.cli.prepare_dataset import prepare_dataset
from donglao_tts.data.compiled import compile_dataset
from donglao_tts.data.dataset import CompiledTTSDataset, MultiCorpusTTSDataset
from donglao_tts.data.sampler import LengthBucketBatchSampler
from scripts.raw_to_compiled import build_staging_manifest, raw_to_compiled

IMPORTERS = Path(__file__).resolve().parents[1] / "scripts" / "raw_to_compiled"
sys.path.insert(0, str(IMPORTERS))


class _FakeCodec:
    def encode_file(self, path):
        if path.endswith("bad.wav"):
            raise ValueError("unreadable")
        return torch.tensor([[1, 2], [3, 4]], dtype=torch.long)


class _FakePipeline:
    def phonemize_batch(self, texts):
        return [f"phoneme:{text}" for text in texts]


class _FakeSentencePiece:
    def __init__(self, model_file):
        self.model_file = model_file

    def encode(self, text, out_type=int):
        return [ord(character) % 127 + 1 for character in text]


def test_atomic_text_writer_preserves_existing_file_on_failure(tmp_path):
    output = tmp_path / "output.txt"
    output.write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with atomic_text_writer(output) as stream:
            stream.write("partial")
            raise RuntimeError("stop")

    assert output.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".donglao-*.tmp"))


def test_prepare_dataset_writes_manifest_and_counts_skips(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "audio_path|speaker_id|text\n"
        "audio/good.wav|speaker-1|Xin chào\n"
        "audio/bad.wav|speaker-2|Bỏ qua\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"

    written, skipped = prepare_dataset(metadata, output, _FakeCodec(), audio_root=tmp_path)

    assert (written, skipped) == (1, 1)
    entry = json.loads(output.read_text(encoding="utf-8"))
    assert entry == {
        "id": 0,
        "source_id": "audio/good.wav",
        "speaker": "speaker-1",
        "text": "Xin chào",
        "codec": [[1, 3], [2, 4]],
    }


def test_prepare_dataset_strict_failure_does_not_publish_partial_output(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "audio_path|speaker_id|text\n"
        "good.wav|speaker-1|Good\n"
        "bad.wav|speaker-1|Bad\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    output.write_text("previous\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to encode"):
        prepare_dataset(metadata, output, _FakeCodec(), audio_root=tmp_path, strict=True)

    assert output.read_text(encoding="utf-8") == "previous\n"


def test_phonemize_manifest_writes_atomic_output(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps({"id": 1, "text": "Xin Chào!"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"

    count = phonemize_manifest(source, output, _FakePipeline())

    assert count == 1
    entry = json.loads(output.read_text(encoding="utf-8"))
    assert entry["phoneme"] == "phoneme:Xin Chào!"


def test_build_phoneme_corpus_accepts_multiple_manifests(tmp_path, monkeypatch):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"text": "Một"}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"text": "Hai"}) + "\n", encoding="utf-8")
    output = tmp_path / "corpus.txt"

    monkeypatch.setattr(
        "donglao_tts.cli.build_phoneme_corpus.Pipeline",
        lambda: _FakePipeline(),
    )

    count = build_phoneme_corpus([(first, "vi"), (second, "vi")], output)

    assert count == 2
    assert output.read_text(encoding="utf-8").splitlines() == [
        "phoneme:Một",
        "phoneme:Hai",
    ]


def test_initialize_config_writes_packaged_template_without_overwriting(tmp_path):
    output = tmp_path / "local.yaml"

    initialize_config(output)

    content = output.read_text(encoding="utf-8")
    assert "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano" in content
    assert "revision:" in content
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        initialize_config(output)


def test_compile_dataset_and_read_memory_mapped_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "donglao_tts.data.compiled.spm.SentencePieceProcessor", _FakeSentencePiece
    )
    monkeypatch.setattr(
        "donglao_tts.data.dataset.spm.SentencePieceProcessor", _FakeSentencePiece
    )
    tokenizer = tmp_path / "spm.model"
    tokenizer.write_bytes(b"stable-tokenizer")
    manifest = tmp_path / "input.jsonl"
    entries = [
        {
            "id": index,
            "speaker": "speaker-1",
            "text": f"text {index}",
            "phoneme": f"phoneme {index}",
            "codec": [[index, index + 1], [index + 2, index + 3]],
        }
        for index in range(3)
    ]
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    output = tmp_path / "compiled"

    catalog = compile_dataset(
        [(manifest, "test-corpus", "vi")],
        output,
        tokenizer,
        shard_size=2,
        val_ratio=0,
        num_quantizers=2,
        codebook_size=16,
    )

    assert len(catalog["shards"]) == 2
    assert catalog["corpus"] == "test-corpus"
    assert sum(shard["rows"] for shard in catalog["shards"]) == 3
    codec = np.load(output / catalog["shards"][0]["codec"], mmap_mode="r")
    assert codec.dtype == np.uint16
    assert codec.shape == (4, 2)

    dataset = CompiledTTSDataset(output, tokenizer, split="train")
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["target_codec"].dtype == torch.uint16
    assert sample["target_codec"].tolist() == entries[0]["codec"]
    assert sample["target_text_ids"].dtype == torch.int32
    combined = MultiCorpusTTSDataset([output], tokenizer, split="train")
    assert len(combined) == len(dataset)
    assert combined.corpus_counts == {"test-corpus": 3}


def test_compile_dataset_append_does_not_rewrite_old_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "donglao_tts.data.compiled.spm.SentencePieceProcessor", _FakeSentencePiece
    )
    tokenizer = tmp_path / "spm.model"
    tokenizer.write_bytes(b"stable-tokenizer")
    output = tmp_path / "compiled"

    def write_manifest(path, utterance_id):
        path.write_text(
            json.dumps(
                {
                    "id": utterance_id,
                    "speaker": "speaker",
                    "text": "text",
                    "phoneme": "phoneme",
                    "codec": [[1, 2]],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_manifest(first, "first")
    write_manifest(second, "second")
    compile_dataset(
        [(first, "corpus-a", "vi")],
        output,
        tokenizer,
        val_ratio=0,
        num_quantizers=2,
        codebook_size=16,
    )
    old_codec = output / "shards/shard-00000/codec.npy"
    old_mtime = old_codec.stat().st_mtime_ns

    catalog = compile_dataset(
        [(second, "corpus-a", "vi")],
        output,
        tokenizer,
        val_ratio=0,
        num_quantizers=2,
        codebook_size=16,
        append=True,
    )

    assert len(catalog["shards"]) == 2
    assert old_codec.stat().st_mtime_ns == old_mtime
    assert (output / "shards/shard-00001/codec.npy").is_file()


def test_compile_dataset_rejects_mixed_corpora(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "donglao_tts.data.compiled.spm.SentencePieceProcessor", _FakeSentencePiece
    )
    tokenizer = tmp_path / "spm.model"
    tokenizer.write_bytes(b"stable-tokenizer")
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": 1,
                "speaker": "speaker",
                "text": "text",
                "phoneme": "phoneme",
                "codec": [[1, 2]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only one corpus"):
        compile_dataset(
            [(manifest, "corpus-a", "vi"), (manifest, "corpus-b", "vi")],
            tmp_path / "compiled",
            tokenizer,
            num_quantizers=2,
            codebook_size=16,
        )


def test_length_sampler_limits_total_codec_frames():
    class _Dataset:
        lengths = [30, 40, 50, 60]

        def __len__(self):
            return len(self.lengths)

        def sequence_length(self, index):
            return self.lengths[index]

        def frame_length(self, index):
            return self.lengths[index]

    dataset = _Dataset()
    sampler = LengthBucketBatchSampler(
        dataset,
        batch_size=4,
        max_frames_per_batch=100,
        bucket_size=4,
        shuffle=False,
    )

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert all(sum(dataset.frame_length(index) for index in batch) <= 100 for batch in batches)


def test_raw_to_compiled_script_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "donglao_tts.data.compiled.spm.SentencePieceProcessor", _FakeSentencePiece
    )
    tokenizer = tmp_path / "spm.model"
    tokenizer.write_bytes(b"stable-tokenizer")
    config = tmp_path / "config.yaml"
    config.write_text(
        "codec:\n"
        "  num_quantizers: 2\n"
        "tokenizer:\n"
        f"  model_path: {tokenizer}\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "audio_path|speaker_id|text\n"
        "audio/one.wav|speaker-1|Xin chào\n"
        "audio/two.wav|speaker-1|Tạm biệt\n",
        encoding="utf-8",
    )
    staging = tmp_path / "staging.jsonl"
    output = tmp_path / "compiled"

    catalog = raw_to_compiled(
        config_path=config,
        metadata_path=metadata,
        output_path=output,
        corpus="portable-test",
        language="vi",
        audio_root=tmp_path,
        staging_path=staging,
        codec=_FakeCodec(),
        pipeline=_FakePipeline(),
        shard_size=1,
        val_ratio=0,
        codebook_size=16,
    )

    assert catalog["corpus"] == "portable-test"
    assert len(catalog["shards"]) == 2
    assert len(staging.read_text(encoding="utf-8").splitlines()) == 2

    stats = build_staging_manifest(
        metadata,
        staging,
        _FakeCodec(),
        _FakePipeline(),
        audio_root=tmp_path,
        resume=True,
        corpus="portable-test",
    )
    assert stats == {
        "written": 0,
        "skipped": 0,
        "already_done": 2,
        "staging_path": str(staging),
    }


def test_phoaudiobook_streaming_converter_resumes(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    from convert_phoaudiobook import convert_phoaudiobook

    monkeypatch.setattr(
        "donglao_tts.data.compiled.spm.SentencePieceProcessor", _FakeSentencePiece
    )

    class _WaveCodec:
        num_channels = 1
        sampling_rate = 16000

        def encode(self, waveform):
            frames = max(1, waveform.shape[-1] // 320)
            return torch.arange(frames * 2, dtype=torch.long).reshape(2, frames) % 16

    def audio_bytes(frequency):
        stream = io.BytesIO()
        time = np.arange(1600, dtype=np.float32) / 16000
        samples = 0.1 * np.sin(2 * np.pi * frequency * time)
        sf.write(stream, samples, 16000, format="WAV")
        return stream.getvalue()

    rows = [
        {
            "audio": {"bytes": audio_bytes(220 + index * 20), "path": None},
            "text": f"Câu số {index}",
            "speaker": "speaker-1",
        }
        for index in range(3)
    ]
    tokenizer = tmp_path / "spm.model"
    tokenizer.write_bytes(b"stable-tokenizer")
    config = tmp_path / "config.yaml"
    config.write_text(
        "codec:\n"
        "  num_quantizers: 2\n"
        "tokenizer:\n"
        f"  model_path: {tokenizer}\n",
        encoding="utf-8",
    )
    output = tmp_path / "compiled"
    work = tmp_path / "work"

    first = convert_phoaudiobook(
        config_path=config,
        output_path=output,
        work_dir=work,
        dataset=rows[:2],
        resolved_revision="test-commit",
        codec=_WaveCodec(),
        pipeline=_FakePipeline(),
        shard_size=2,
        val_ratio=0,
        codebook_size=16,
    )
    resumed = convert_phoaudiobook(
        config_path=config,
        output_path=output,
        work_dir=work,
        dataset=rows[2:],
        resolved_revision="test-commit",
        codec=_WaveCodec(),
        pipeline=_FakePipeline(),
        shard_size=2,
        val_ratio=0,
        codebook_size=16,
        resume=True,
    )

    assert first["compiled"] == 2
    assert resumed["compiled"] == 3
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["corpus"] == "phoaudiobook"
    assert sum(shard["rows"] for shard in catalog["shards"]) == 3
