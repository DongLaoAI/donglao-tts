import json
import os
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
from donglao_g2p import Pipeline, __phoneme_profile__

from donglao_tts.data.compiled import FORMAT_NAME, FORMAT_VERSION, ROW_DTYPE, _hash_file
from donglao_tts.data.dataset import CompiledTTSDataset


CHANGE_PHONEME = Path(__file__).resolve().parents[1] / "scripts" / "change_phoneme"
sys.path.insert(0, str(CHANGE_PHONEME))
import change_phoneme  # noqa: E402


def _train_tokenizer(corpus, prefix):
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=64,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
    )
    return prefix.with_suffix(".model")


def _compiled_fixture(root):
    dataset = root / "sample"
    shard = dataset / "shards" / "shard-00000"
    shard.mkdir(parents=True)
    corpus = root / "old-phonemes.txt"
    corpus.write_text("old phone one\nold phone two\n", encoding="utf-8")
    old_tokenizer = _train_tokenizer(corpus, root / "old-spm")
    tokenizer = spm.SentencePieceProcessor(model_file=str(old_tokenizer))
    old_ids = [tokenizer.encode(text, out_type=int) for text in ("old phone one", "old phone two")]

    codec = np.arange(24, dtype=np.uint16).reshape(12, 2)
    np.save(shard / "codec.npy", codec)
    np.save(shard / "text_ids.npy", np.asarray(old_ids[0] + old_ids[1], dtype=np.int32))
    rows = np.zeros(2, dtype=ROW_DTYPE)
    rows[0] = (1, 11, 0, 5, 0, len(old_ids[0]), 0)
    rows[1] = (2, 11, 5, 7, len(old_ids[0]), len(old_ids[1]), 1)
    np.save(shard / "rows.npy", rows)

    entries = [
        {
            "utterance_id": "sample:one",
            "corpus": "sample",
            "source_id": "one",
            "speaker_id": "speaker",
            "speaker_uid": "sample:speaker",
            "language": "vi",
            "split": "train",
            "text": "Xin chào.",
            "phoneme": "old phone one",
            "codec_frames": 5,
            "text_tokens": len(old_ids[0]),
        },
        {
            "utterance_id": "sample:two",
            "corpus": "sample",
            "source_id": "two",
            "speaker_id": "speaker",
            "speaker_uid": "sample:speaker",
            "language": "en",
            "split": "val",
            "text": "Hello world.",
            "phoneme": "old phone two",
            "codec_frames": 7,
            "text_tokens": len(old_ids[1]),
        },
    ]
    with (shard / "metadata.jsonl").open("w", encoding="utf-8") as destination:
        for entry in entries:
            destination.write(json.dumps(entry, ensure_ascii=False) + "\n")
    catalog = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "corpus": "sample",
        "language": "mixed",
        "tokenizer": {"sha256": _hash_file(old_tokenizer)},
        "codec": {"dtype": "uint16", "num_quantizers": 2, "codebook_size": 1024},
        "split": {"seed": 42, "val_ratio": 0.5},
        "shards": [
            {
                "name": "shard-00000",
                "rows": 2,
                "codec_frames": 12,
                "text_tokens": sum(map(len, old_ids)),
                "split_counts": {"train": 1, "val": 1},
                "corpus": "sample",
                "codec": "shards/shard-00000/codec.npy",
                "text_ids": "shards/shard-00000/text_ids.npy",
                "rows_index": "shards/shard-00000/rows.npy",
                "metadata": "shards/shard-00000/metadata.jsonl",
            }
        ],
    }
    (dataset / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    return dataset, old_tokenizer


def test_prepare_and_finalize_rephonemized_dataset(tmp_path):
    compiled = tmp_path / "compiled"
    source_dataset, old_tokenizer = _compiled_fixture(compiled)
    work = tmp_path / "work"
    phoneme_corpus = tmp_path / "phonemes-v1.txt"

    summary = change_phoneme.prepare(
        compiled,
        work,
        phoneme_corpus,
        batch_size=2,
        on_error="fail",
        resume=True,
    )

    phonemes = phoneme_corpus.read_text(encoding="utf-8").splitlines()
    assert summary["written"] == 2
    assert summary["skipped"] == 0
    assert phonemes == Pipeline(language="auto").phonemize_batch(
        ["Xin chào.", "Hello world."]
    )

    new_tokenizer = _train_tokenizer(phoneme_corpus, tmp_path / "new-spm")
    output = tmp_path / "complied_v1"
    result = change_phoneme.finalize(
        compiled,
        work,
        new_tokenizer,
        output,
        resume=True,
    )

    rebuilt = output / "sample"
    catalog = json.loads((rebuilt / "catalog.json").read_text())
    assert catalog["tokenizer"]["sha256"] == _hash_file(new_tokenizer)
    assert catalog["tokenizer"]["sha256"] != _hash_file(old_tokenizer)
    assert catalog["g2p"]["package"] == "donglao-g2p"
    assert catalog["g2p"]["phoneme_profile"] == __phoneme_profile__
    assert result["datasets"] == [str(rebuilt)]

    source_codec = source_dataset / "shards" / "shard-00000" / "codec.npy"
    rebuilt_codec = rebuilt / "shards" / "shard-00000" / "codec.npy"
    assert np.array_equal(np.load(source_codec), np.load(rebuilt_codec))
    assert os.stat(source_codec).st_ino == os.stat(rebuilt_codec).st_ino

    rows = np.load(rebuilt / "shards" / "shard-00000" / "rows.npy")
    text_ids = np.load(rebuilt / "shards" / "shard-00000" / "text_ids.npy")
    tokenizer = spm.SentencePieceProcessor(model_file=str(new_tokenizer))
    expected = [tokenizer.encode(phoneme, out_type=int) for phoneme in phonemes]
    assert rows[0]["text_offset"] == 0
    assert rows[0]["text_length"] == len(expected[0])
    assert rows[1]["text_offset"] == len(expected[0])
    assert rows[1]["text_length"] == len(expected[1])
    assert text_ids.tolist() == expected[0] + expected[1]

    metadata = [
        json.loads(line)
        for line in (
            rebuilt / "shards" / "shard-00000" / "metadata.jsonl"
        ).read_text().splitlines()
    ]
    assert [entry["phoneme"] for entry in metadata] == phonemes

    training_dataset = CompiledTTSDataset(
        rebuilt,
        str(new_tokenizer),
        split="train",
    )
    sample = training_dataset[0]
    assert sample["target_text_ids"].tolist() == expected[0]
    assert sample["target_codec"].shape == (5, 2)
