#!/usr/bin/env python3
"""Decode RVQ samples from compiled DongLao TTS datasets for manual inspection.

This is intentionally a no-CLI utility. Edit the values in the ``__main__``
block when a different dataset, output directory, or sample count is needed.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from donglao_tts.data.compiled import FORMAT_NAME, FORMAT_VERSION  # noqa: E402


def _catalog_paths(data_path):
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"compiled dataset does not exist: {data_path}")
    if data_path.is_file():
        if data_path.name != "catalog.json":
            raise ValueError(f"expected catalog.json, got: {data_path}")
        return [data_path]

    direct_catalog = data_path / "catalog.json"
    if direct_catalog.is_file():
        return [direct_catalog]

    catalogs = sorted(data_path.rglob("catalog.json"))
    if not catalogs:
        raise FileNotFoundError(f"no catalog.json found under: {data_path}")
    return catalogs


def _load_catalog(catalog_path):
    with catalog_path.open("r", encoding="utf-8") as source:
        catalog = json.load(source)
    if catalog.get("format") != FORMAT_NAME or catalog.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported compiled dataset format: {catalog_path}")
    if not isinstance(catalog.get("shards"), list):
        raise ValueError(f"catalog has no shards list: {catalog_path}")
    return catalog


def _dataset_file(dataset_root, relative_path):
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        raise ValueError(f"absolute shard path is not allowed: {relative_path}")
    resolved = (dataset_root / relative_path).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"shard path escapes dataset root: {relative_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing compiled dataset file: {resolved}")
    return resolved


def _safe_name(value):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name[:80] or "sample"


def _write_text_file(path, metadata, rvq_shape):
    path.write_text(
        "\n".join(
            [
                f"utterance_id: {metadata.get('utterance_id', '')}",
                f"corpus: {metadata.get('corpus', '')}",
                f"language: {metadata.get('language', '')}",
                f"speaker_id: {metadata.get('speaker_id', '')}",
                f"split: {metadata.get('split', '')}",
                f"rvq_shape: {list(rvq_shape)}",
                "",
                "text:",
                str(metadata.get("text", "")),
                "",
                "phoneme:",
                str(metadata.get("phoneme", "")),
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_compiled_rvq(
    data_path,
    output_path,
    config_path,
    *,
    samples_per_dataset=5,
    codec=None,
):
    """Decode the first N samples of every compiled dataset.

    Args:
        data_path: A compiled dataset directory, its ``catalog.json``, or a
            parent directory containing multiple compiled datasets.
        output_path: Directory receiving paired ``.wav`` and ``.txt`` files.
        config_path: DongLao TTS YAML config used to construct MOSS Codec.
        samples_per_dataset: Maximum samples decoded from each catalog.
        codec: Optional already-created codec, useful for testing.

    Returns:
        A list of manifest dictionaries describing the exported samples.
    """
    if samples_per_dataset < 1:
        raise ValueError("samples_per_dataset must be at least 1")

    output_root = Path(output_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if codec is None:
        # Keep the heavyweight Transformers import out of module import time and
        # only load the pretrained codec when real decoding is requested.
        from donglao_tts.models.codec.moss_codec import MossCodec

        codec = MossCodec.from_config(Path(config_path).expanduser().resolve())

    exported = []
    for catalog_path in _catalog_paths(data_path):
        dataset_root = catalog_path.parent.resolve()
        catalog = _load_catalog(catalog_path)
        corpus = _safe_name(catalog.get("corpus", dataset_root.name))
        expected_quantizers = int(catalog["codec"]["num_quantizers"])
        if expected_quantizers != codec.num_quantizers:
            raise ValueError(
                f"{catalog_path}: dataset has {expected_quantizers} quantizers, "
                f"but codec uses {codec.num_quantizers}"
            )

        dataset_output = output_root / corpus
        dataset_output.mkdir(parents=True, exist_ok=True)
        dataset_count = 0

        for shard in catalog["shards"]:
            if dataset_count >= samples_per_dataset:
                break
            rows_path = _dataset_file(dataset_root, shard["rows_index"])
            codec_path = _dataset_file(dataset_root, shard["codec"])
            metadata_path = _dataset_file(dataset_root, shard["metadata"])
            rows = np.load(rows_path, mmap_mode="r")
            rvq_storage = np.load(codec_path, mmap_mode="r")

            with metadata_path.open("r", encoding="utf-8") as metadata_source:
                for row_index, (row, metadata_line) in enumerate(zip(rows, metadata_source)):
                    if dataset_count >= samples_per_dataset:
                        break
                    metadata = json.loads(metadata_line)
                    frame_start = int(row["codec_offset"])
                    frame_end = frame_start + int(row["codec_length"])
                    rvq = np.asarray(rvq_storage[frame_start:frame_end], dtype=np.int64)
                    if rvq.ndim != 2 or rvq.shape[1] != expected_quantizers:
                        raise ValueError(
                            f"invalid RVQ shape at {rows_path}:{row_index}: {rvq.shape}"
                        )

                    # Compiled storage is [T, num_quantizers]; MOSS decode expects
                    # [num_quantizers, T].
                    rvq_codes = torch.from_numpy(rvq.T.copy()).long()
                    waveform = codec.decode(rvq_codes)

                    sequence = dataset_count + 1
                    source_name = _safe_name(
                        metadata.get("source_id", metadata.get("utterance_id", sequence))
                    )
                    stem = f"{sequence:03d}_{source_name}"
                    audio_path = dataset_output / f"{stem}.wav"
                    text_path = dataset_output / f"{stem}.txt"
                    codec.save_audio(waveform, str(audio_path))
                    _write_text_file(text_path, metadata, rvq_codes.shape)

                    record = {
                        "corpus": catalog.get("corpus", corpus),
                        "language": metadata.get("language", catalog.get("language", "")),
                        "utterance_id": metadata.get("utterance_id", ""),
                        "speaker_id": metadata.get("speaker_id", ""),
                        "text": metadata.get("text", ""),
                        "phoneme": metadata.get("phoneme", ""),
                        "codec_frames": int(row["codec_length"]),
                        "audio": str(audio_path.relative_to(output_root)),
                        "text_file": str(text_path.relative_to(output_root)),
                    }
                    exported.append(record)
                    dataset_count += 1
                    print(
                        f"[{corpus}] {dataset_count}/{samples_per_dataset}: "
                        f"{audio_path.name}"
                    )

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as destination:
        for record in exported:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(exported)} samples to: {output_root}")
    print(f"Manifest: {manifest_path}")
    return exported


if __name__ == "__main__":
    export_compiled_rvq(
        data_path=PROJECT_ROOT / "DATASET" / "complied_v1/emilia_en",
        output_path=PROJECT_ROOT / "tests" / "codec_rvq_output",
        config_path=PROJECT_ROOT / "configs" / "base.yaml",
        samples_per_dataset=5,
    )
