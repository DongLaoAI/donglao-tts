"""Compiled, memory-mapped dataset format used by the training dataloader."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import sentencepiece as spm

from donglao_tts.cli._io import atomic_text_writer


FORMAT_NAME = "donglao-tts-compiled"
FORMAT_VERSION = 2
SPLIT_NAMES = ("train", "val")
ROW_DTYPE = np.dtype(
    [
        ("utterance_key", "<u8"),
        ("speaker_key", "<u8"),
        ("codec_offset", "<u8"),
        ("codec_length", "<u4"),
        ("text_offset", "<u8"),
        ("text_length", "<u4"),
        ("split", "u1"),
    ]
)


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_u64(value):
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little"
    )


def _split_id(utterance_id, val_ratio, seed):
    digest = hashlib.blake2b(
        f"{seed}:{utterance_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int(int.from_bytes(digest, "little") % 1_000_000 < val_ratio * 1_000_000)


def _safe_corpus_name(name):
    if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(
            f"invalid corpus name {name!r}; use letters, digits, dot, underscore, or hyphen"
        )
    return name


def _load_existing_ids(root, catalog):
    ids = set()
    for shard in catalog.get("shards", []):
        metadata_path = root / shard["metadata"]
        with metadata_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    utterance_id = json.loads(line)["utterance_id"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"invalid compiled metadata at {metadata_path}:{line_number}"
                    ) from exc
                if utterance_id in ids:
                    raise ValueError(f"duplicate utterance_id in compiled dataset: {utterance_id}")
                ids.add(utterance_id)
    return ids


def _validate_entry(entry, manifest_path, line_number, corpus, num_quantizers, codebook_size):
    for field in ("id", "speaker", "text", "phoneme", "codec"):
        if field not in entry:
            raise ValueError(f"missing {field!r} at {manifest_path}:{line_number}")

    source_id = str(entry.get("source_id", entry["id"]))
    utterance_id = f"{corpus}:{source_id}"
    speaker_id = str(entry["speaker"])
    if not speaker_id:
        raise ValueError(f"empty speaker at {manifest_path}:{line_number}")
    if not str(entry["phoneme"]).strip():
        raise ValueError(f"empty phoneme at {manifest_path}:{line_number}")

    codec = np.asarray(entry["codec"])
    if codec.ndim != 2 or codec.shape[0] == 0 or codec.shape[1] != num_quantizers:
        raise ValueError(
            f"codec at {manifest_path}:{line_number} must have shape [T, {num_quantizers}], "
            f"got {codec.shape}"
        )
    if not np.issubdtype(codec.dtype, np.integer):
        raise ValueError(f"codec at {manifest_path}:{line_number} must contain integers")
    if int(codec.min()) < 0 or int(codec.max()) >= codebook_size:
        raise ValueError(
            f"codec at {manifest_path}:{line_number} must be in [0, {codebook_size})"
        )
    return utterance_id, speaker_id, codec.astype(np.uint16, copy=False)


def _write_shard(
    temporary_root,
    shard_index,
    records,
    tokenizer,
    val_ratio,
    seed,
    num_quantizers,
    codebook_size,
):
    shard_name = f"shard-{shard_index:05d}"
    shard_dir = temporary_root / shard_name
    shard_dir.mkdir()

    prepared = []
    total_frames = 0
    total_tokens = 0
    split_counts = [0, 0]
    for record in records:
        entry = record["entry"]
        utterance_id, speaker_id, codec = _validate_entry(
            entry,
            record["manifest_path"],
            record["line_number"],
            record["corpus"],
            num_quantizers,
            codebook_size,
        )
        text_ids = np.asarray(tokenizer.encode(entry["phoneme"], out_type=int), dtype=np.int32)
        if text_ids.ndim != 1 or text_ids.size == 0:
            raise ValueError(
                f"tokenizer produced no tokens at "
                f"{record['manifest_path']}:{record['line_number']}"
            )
        if int(text_ids.min()) < 0:
            raise ValueError(
                f"tokenizer produced a negative id at "
                f"{record['manifest_path']}:{record['line_number']}"
            )
        split = _split_id(utterance_id, val_ratio, seed)
        prepared.append((record, utterance_id, speaker_id, codec, text_ids, split))
        total_frames += len(codec)
        total_tokens += len(text_ids)
        split_counts[split] += 1

    codec_out = np.lib.format.open_memmap(
        shard_dir / "codec.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(total_frames, num_quantizers),
    )
    text_out = np.lib.format.open_memmap(
        shard_dir / "text_ids.npy", mode="w+", dtype=np.int32, shape=(total_tokens,)
    )
    rows_out = np.lib.format.open_memmap(
        shard_dir / "rows.npy", mode="w+", dtype=ROW_DTYPE, shape=(len(prepared),)
    )

    codec_offset = 0
    text_offset = 0
    with (shard_dir / "metadata.jsonl").open("w", encoding="utf-8") as metadata:
        for row_index, item in enumerate(prepared):
            record, utterance_id, speaker_id, codec, text_ids, split = item
            codec_end = codec_offset + len(codec)
            text_end = text_offset + len(text_ids)
            codec_out[codec_offset:codec_end] = codec
            text_out[text_offset:text_end] = text_ids
            rows_out[row_index] = (
                _stable_u64(utterance_id),
                _stable_u64(f"{record['corpus']}:{speaker_id}"),
                codec_offset,
                len(codec),
                text_offset,
                len(text_ids),
                split,
            )
            metadata.write(
                json.dumps(
                    {
                        "utterance_id": utterance_id,
                        "corpus": record["corpus"],
                        "source_id": str(
                            record["entry"].get("source_id", record["entry"]["id"])
                        ),
                        "speaker_id": speaker_id,
                        "speaker_uid": f"{record['corpus']}:{speaker_id}",
                        "language": record["language"],
                        "split": SPLIT_NAMES[split],
                        "text": record["entry"]["text"],
                        "phoneme": record["entry"]["phoneme"],
                        "codec_frames": len(codec),
                        "text_tokens": len(text_ids),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            codec_offset = codec_end
            text_offset = text_end

    codec_out.flush()
    text_out.flush()
    rows_out.flush()
    del codec_out, text_out, rows_out

    return {
        "name": shard_name,
        "rows": len(prepared),
        "codec_frames": total_frames,
        "text_tokens": total_tokens,
        "split_counts": {"train": split_counts[0], "val": split_counts[1]},
        "corpus": records[0]["corpus"],
        "codec": f"shards/{shard_name}/codec.npy",
        "text_ids": f"shards/{shard_name}/text_ids.npy",
        "rows_index": f"shards/{shard_name}/rows.npy",
        "metadata": f"shards/{shard_name}/metadata.jsonl",
    }


def compile_dataset(
    manifests,
    output_path,
    spm_model_path,
    *,
    shard_size=4096,
    val_ratio=0.01,
    seed=42,
    num_quantizers=8,
    codebook_size=1024,
    append=False,
    check_existing_ids=True,
):
    """Compile phonemized JSONL manifests into immutable, memory-mapped shards.

    ``manifests`` is an iterable of ``(path, corpus_name, language)`` tuples.
    Existing shards are never modified. With ``append=True``, new shards are published by an
    atomic catalog update after every new file has been validated and moved into place.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if num_quantizers < 1 or codebook_size < 1 or codebook_size > 65536:
        raise ValueError("invalid codec dimensions for uint16 storage")

    root = Path(output_path).resolve()
    tokenizer_path = Path(spm_model_path).resolve()
    tokenizer_sha256 = _hash_file(tokenizer_path)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    normalized_manifests = [
        (Path(path).resolve(), _safe_corpus_name(corpus), language)
        for path, corpus, language in manifests
    ]
    if not normalized_manifests:
        raise ValueError("at least one manifest is required")
    corpora = {corpus for _, corpus, _ in normalized_manifests}
    languages = {language for _, _, language in normalized_manifests}
    if len(corpora) != 1 or len(languages) != 1:
        raise ValueError(
            "one compiled dataset may contain only one corpus and language; "
            "use a separate --output directory for each corpus"
        )
    corpus = next(iter(corpora))
    language = next(iter(languages))

    if append:
        catalog_path = root / "catalog.json"
        if not catalog_path.is_file():
            raise FileNotFoundError(f"cannot append: missing {catalog_path}")
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog.get("format") != FORMAT_NAME or catalog.get("version") != FORMAT_VERSION:
            raise ValueError("cannot append to an incompatible compiled dataset")
        expected = (
            catalog["corpus"],
            catalog["language"],
            catalog["tokenizer"]["sha256"],
            catalog["codec"]["num_quantizers"],
            catalog["codec"]["codebook_size"],
            catalog["split"]["seed"],
            catalog["split"]["val_ratio"],
        )
        actual = (
            corpus,
            language,
            tokenizer_sha256,
            num_quantizers,
            codebook_size,
            seed,
            val_ratio,
        )
        if actual != expected:
            raise ValueError(
                "tokenizer, codec, or split configuration does not match the dataset"
            )
        # Streaming importers may already load/filter the existing IDs once at startup. Let them
        # avoid an O(total_rows) rescan for every appended shard while retaining safe validation
        # as the default for all normal callers.
        known_ids = _load_existing_ids(root, catalog) if check_existing_ids else set()
    else:
        if root.exists():
            raise FileExistsError(f"refusing to overwrite compiled dataset: {root}")
        catalog = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "corpus": corpus,
            "language": language,
            "tokenizer": {"sha256": tokenizer_sha256},
            "codec": {
                "dtype": "uint16",
                "num_quantizers": num_quantizers,
                "codebook_size": codebook_size,
            },
            "split": {"seed": seed, "val_ratio": val_ratio},
            "shards": [],
        }
        known_ids = set()

    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".donglao-compile-", dir=parent))
    temporary_shards = temporary_root / "shards"
    temporary_shards.mkdir()
    new_shards = []
    shard_index = len(catalog["shards"])

    try:
        for manifest_path, corpus, language in normalized_manifests:
            records = []
            with manifest_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSON at {manifest_path}:{line_number}"
                        ) from exc
                    source_id = entry.get("source_id", entry.get("id"))
                    utterance_id = f"{corpus}:{source_id}"
                    if utterance_id in known_ids:
                        raise ValueError(f"duplicate utterance_id: {utterance_id}")
                    known_ids.add(utterance_id)
                    records.append(
                        {
                            "entry": entry,
                            "manifest_path": str(manifest_path),
                            "line_number": line_number,
                            "corpus": corpus,
                            "language": language,
                        }
                    )
                    if len(records) == shard_size:
                        new_shards.append(
                            _write_shard(
                                temporary_shards,
                                shard_index,
                                records,
                                tokenizer,
                                val_ratio,
                                seed,
                                num_quantizers,
                                codebook_size,
                            )
                        )
                        shard_index += 1
                        records = []
                if records:
                    new_shards.append(
                        _write_shard(
                            temporary_shards,
                            shard_index,
                            records,
                            tokenizer,
                            val_ratio,
                            seed,
                            num_quantizers,
                            codebook_size,
                        )
                    )
                    shard_index += 1

        if not new_shards:
            raise ValueError("the input manifests contain no utterances")

        if append:
            destination_shards = root / "shards"
            destination_shards.mkdir(exist_ok=True)
            for shard in new_shards:
                os.replace(
                    temporary_shards / shard["name"], destination_shards / shard["name"]
                )
        else:
            catalog["shards"] = new_shards
            with (temporary_root / "catalog.json").open("w", encoding="utf-8") as destination:
                json.dump(catalog, destination, ensure_ascii=False, indent=2)
                destination.write("\n")
            os.replace(temporary_root, root)
            temporary_root = None
            return catalog

        catalog["shards"].extend(new_shards)
        with atomic_text_writer(root / "catalog.json") as destination:
            json.dump(catalog, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        return catalog
    finally:
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)


def load_catalog(root_path):
    root = Path(root_path).resolve()
    catalog_path = root / "catalog.json"
    with catalog_path.open("r", encoding="utf-8") as source:
        catalog = json.load(source)
    if catalog.get("format") != FORMAT_NAME or catalog.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported compiled dataset format in {catalog_path}")
    return root, catalog
