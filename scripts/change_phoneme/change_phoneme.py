#!/usr/bin/env python3
"""Re-phonemize compiled datasets and rebuild their SentencePiece text indexes."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import sentencepiece as spm
from donglao_g2p import Pipeline, __phoneme_profile__, __version__


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PACKAGE = _SOURCE_ROOT / "src"
if _SOURCE_PACKAGE.is_dir():
    sys.path.insert(0, str(_SOURCE_PACKAGE))

from donglao_tts.data.compiled import FORMAT_NAME, ROW_DTYPE, _hash_file  # noqa: E402


MINIMUM_G2P_VERSION = (0, 3)


def _require_supported_g2p():
    try:
        version = tuple(int(part) for part in __version__.split(".")[:2])
    except (AttributeError, ValueError) as exc:
        raise RuntimeError(f"invalid donglao-g2p version: {__version__!r}") from exc
    if version < MINIMUM_G2P_VERSION:
        raise RuntimeError(
            f"donglao-g2p >=0.3,<0.4 is required, found {__version__}; "
            "run `python -m pip install --upgrade \"donglao-g2p>=0.3,<0.4\"`"
        )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _atomic_text(path, *, newline=None):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline=newline,
        ) as destination:
            yield destination
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path, value):
    with _atomic_text(path) as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def _discover_catalogs(input_path):
    input_path = Path(input_path).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"compiled input directory does not exist: {input_path}")
    direct = input_path / "catalog.json"
    if direct.is_file():
        return input_path.parent, [direct]
    catalogs = sorted(input_path.rglob("catalog.json"))
    if not catalogs:
        raise FileNotFoundError(f"no catalog.json found under {input_path}")
    return input_path, catalogs


def _safe_source_path(dataset_root, relative_path, label):
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        raise ValueError(f"absolute {label} path is not allowed: {relative_path}")
    resolved = (dataset_root / relative_path).resolve()
    try:
        resolved.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes dataset root: {relative_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _new_plan(input_path):
    input_root, catalog_paths = _discover_catalogs(input_path)
    datasets = []
    for catalog_path in catalog_paths:
        catalog = _read_json(catalog_path)
        if catalog.get("format") != FORMAT_NAME:
            raise ValueError(f"unsupported format in {catalog_path}")
        relative_root = catalog_path.parent.relative_to(input_root)
        if str(relative_root) == ".":
            relative_root = Path(catalog.get("corpus") or catalog_path.parent.name)
        shards = []
        for shard in catalog.get("shards", []):
            metadata = _safe_source_path(
                catalog_path.parent,
                shard["metadata"],
                "metadata",
            )
            shards.append(
                {
                    "name": shard["name"],
                    "metadata": str(metadata),
                    "metadata_relative": shard["metadata"],
                    "metadata_size": metadata.stat().st_size,
                }
            )
        datasets.append(
            {
                "relative_root": str(relative_root),
                "source_root": str(catalog_path.parent.resolve()),
                "catalog": str(catalog_path.resolve()),
                "catalog_sha256": _sha256(catalog_path),
                "corpus": catalog["corpus"],
                "language": catalog["language"],
                "shards": shards,
            }
        )
    return {
        "format": "donglao-tts-rephonemize-plan",
        "version": 1,
        "input": str(Path(input_path).resolve()),
        "g2p": {
            "package": "donglao-g2p",
            "version": __version__,
            "phoneme_profile": __phoneme_profile__,
        },
        "datasets": datasets,
    }


def _load_or_create_plan(input_path, work_dir, resume):
    work_dir = Path(work_dir).resolve()
    plan_path = work_dir / "plan.json"
    proposed = _new_plan(input_path)
    if plan_path.is_file():
        existing = _read_json(plan_path)
        if not resume:
            raise FileExistsError(f"work plan exists; pass --resume: {plan_path}")
        if existing != proposed:
            raise ValueError(
                "compiled input or donglao-g2p profile changed since this work plan was "
                f"created; choose a new --work-dir instead of mixing snapshots: {plan_path}"
            )
        return existing
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_json(plan_path, proposed)
    return proposed


def _prepared_metadata_path(work_dir, dataset, shard):
    return (
        Path(work_dir).resolve()
        / "prepared"
        / dataset["relative_root"]
        / shard["metadata_relative"]
    )


def _phonemize_resilient(pipeline, records, on_error):
    texts = [entry["text"] for _, entry in records]
    try:
        phonemes = pipeline.phonemize_batch(texts)
        if len(phonemes) != len(records):
            raise RuntimeError(
                f"donglao-g2p returned {len(phonemes)} values for {len(records)} texts"
            )
        return list(zip(records, phonemes)), []
    except Exception:
        if on_error == "fail":
            raise

    successful = []
    errors = []
    for record in records:
        row_index, entry = record
        try:
            successful.append((record, pipeline.phonemize(entry["text"])))
        except Exception as exc:
            errors.append((row_index, entry.get("utterance_id"), str(exc)))
    return successful, errors


def _prepare_shard(
    source_path,
    destination_path,
    error_path,
    pipeline,
    *,
    batch_size,
    on_error,
    reject_unk,
):
    written = 0
    skipped = 0
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_text(destination_path) as destination:
        with _atomic_text(error_path, newline="") as error_destination:
            error_writer = csv.writer(
                error_destination,
                delimiter="|",
                quotechar='"',
                lineterminator="\n",
            )
            error_writer.writerow(("source_row_index", "utterance_id", "error"))
            batch = []

            def flush():
                nonlocal written, skipped, batch
                if not batch:
                    return
                successful, errors = _phonemize_resilient(pipeline, batch, on_error)
                for row_index, utterance_id, message in errors:
                    error_writer.writerow((row_index, utterance_id or "", message))
                    skipped += 1
                for (row_index, entry), phoneme in successful:
                    try:
                        if not isinstance(phoneme, str) or not phoneme.strip():
                            raise ValueError("donglao-g2p returned an empty phoneme")
                        if reject_unk and "<unk>" in phoneme:
                            raise ValueError("donglao-g2p output contains <unk>")
                        updated = dict(entry)
                        updated["phoneme"] = phoneme.strip()
                        updated["_change_phoneme_source_row"] = row_index
                        destination.write(
                            json.dumps(updated, ensure_ascii=False) + "\n"
                        )
                        written += 1
                    except Exception as exc:
                        if on_error == "fail":
                            raise
                        error_writer.writerow(
                            (row_index, entry.get("utterance_id", ""), str(exc))
                        )
                        skipped += 1
                batch = []

            with Path(source_path).open("r", encoding="utf-8") as source:
                for row_index, line in enumerate(source):
                    try:
                        entry = json.loads(line)
                        if not isinstance(entry.get("text"), str) or not entry["text"].strip():
                            raise ValueError("missing or empty text")
                    except Exception as exc:
                        if on_error == "fail":
                            raise ValueError(
                                f"invalid metadata at {source_path}:{row_index + 1}"
                            ) from exc
                        error_writer.writerow((row_index, "", str(exc)))
                        skipped += 1
                        continue
                    batch.append((row_index, entry))
                    if len(batch) == batch_size:
                        flush()
                flush()
    return {"written": written, "skipped": skipped}


def prepare(
    input_path,
    work_dir,
    phoneme_corpus,
    *,
    batch_size=4096,
    num_threads=None,
    on_error="skip",
    reject_unk=False,
    resume=False,
):
    _require_supported_g2p()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    plan = _load_or_create_plan(input_path, work_dir, resume)
    pipeline = Pipeline(language='auto', num_threads=num_threads)
    total_written = 0
    total_skipped = 0

    for dataset in plan["datasets"]:
        for shard in dataset["shards"]:
            prepared_path = _prepared_metadata_path(work_dir, dataset, shard)
            marker_path = prepared_path.with_suffix(
                prepared_path.suffix + ".done.json"
            )
            error_path = prepared_path.with_suffix(
                prepared_path.suffix + ".errors.csv"
            )
            marker_identity = {
                "source": shard["metadata"],
                "source_size": shard["metadata_size"],
                "g2p": plan["g2p"],
                "reject_unk": reject_unk,
            }
            if marker_path.is_file() and prepared_path.is_file():
                marker = _read_json(marker_path)
                if marker.get("identity") != marker_identity:
                    raise ValueError(
                        f"prepared shard identity changed; use a new --work-dir: {prepared_path}"
                    )
                total_written += marker["written"]
                total_skipped += marker["skipped"]
                continue

            print(f"phonemize: {dataset['corpus']} / {shard['name']}")
            stats = _prepare_shard(
                shard["metadata"],
                prepared_path,
                error_path,
                pipeline,
                batch_size=batch_size,
                on_error=on_error,
                reject_unk=reject_unk,
            )
            _write_json(marker_path, {"identity": marker_identity, **stats})
            total_written += stats["written"]
            total_skipped += stats["skipped"]

    corpus_count = 0
    with _atomic_text(phoneme_corpus) as destination:
        for dataset in plan["datasets"]:
            for shard in dataset["shards"]:
                prepared_path = _prepared_metadata_path(work_dir, dataset, shard)
                with prepared_path.open("r", encoding="utf-8") as source:
                    for line in source:
                        destination.write(json.loads(line)["phoneme"] + "\n")
                        corpus_count += 1
    summary = {
        "written": total_written,
        "skipped": total_skipped,
        "phoneme_corpus": str(Path(phoneme_corpus).resolve()),
        "corpus_lines": corpus_count,
        "g2p": plan["g2p"],
    }
    _write_json(Path(work_dir) / "prepare-summary.json", summary)
    return summary


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _finalize_shard(
    source_root,
    source_shard,
    prepared_path,
    output_root,
    tokenizer,
):
    prepared = []
    with prepared_path.open("r", encoding="utf-8") as source:
        for line in source:
            entry = json.loads(line)
            source_row = int(entry["_change_phoneme_source_row"])
            text_ids = tokenizer.encode(entry["phoneme"], out_type=int)
            if not text_ids or min(text_ids) < 0:
                raise ValueError(
                    f"new tokenizer produced invalid ids for {entry.get('utterance_id')}"
                )
            prepared.append((source_row, entry, np.asarray(text_ids, dtype=np.int32)))

    if not prepared:
        return None
    source_rows_path = _safe_source_path(
        source_root,
        source_shard["rows_index"],
        "rows index",
    )
    source_codec_path = _safe_source_path(
        source_root,
        source_shard["codec"],
        "codec",
    )
    source_rows = np.load(source_rows_path, mmap_mode="r")
    source_indexes = [item[0] for item in prepared]
    if source_indexes != sorted(set(source_indexes)):
        raise ValueError(f"prepared source rows are duplicated or unordered: {prepared_path}")
    if source_indexes[-1] >= len(source_rows):
        raise ValueError(f"prepared source row is outside source shard: {prepared_path}")

    output_codec_path = output_root / source_shard["codec"]
    storage_mode = _link_or_copy(source_codec_path, output_codec_path)
    output_text_path = output_root / source_shard["text_ids"]
    output_rows_path = output_root / source_shard["rows_index"]
    output_metadata_path = output_root / source_shard["metadata"]
    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows_path.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = sum(len(item[2]) for item in prepared)
    text_output = np.lib.format.open_memmap(
        output_text_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_tokens,),
    )
    rows_output = np.lib.format.open_memmap(
        output_rows_path,
        mode="w+",
        dtype=ROW_DTYPE,
        shape=(len(prepared),),
    )
    text_offset = 0
    codec_frames = 0
    split_counts = [0, 0]
    with output_metadata_path.open("w", encoding="utf-8") as metadata_output:
        for output_index, (source_index, entry, text_ids) in enumerate(prepared):
            source_row = source_rows[source_index]
            text_end = text_offset + len(text_ids)
            text_output[text_offset:text_end] = text_ids
            rows_output[output_index] = source_row
            rows_output[output_index]["text_offset"] = text_offset
            rows_output[output_index]["text_length"] = len(text_ids)
            codec_length = int(source_row["codec_length"])
            split = int(source_row["split"])
            codec_frames += codec_length
            split_counts[split] += 1
            clean_entry = {
                key: value
                for key, value in entry.items()
                if not key.startswith("_change_phoneme_")
            }
            clean_entry["codec_frames"] = codec_length
            clean_entry["text_tokens"] = len(text_ids)
            metadata_output.write(
                json.dumps(clean_entry, ensure_ascii=False) + "\n"
            )
            text_offset = text_end
    text_output.flush()
    rows_output.flush()
    del text_output, rows_output, source_rows
    return {
        **source_shard,
        "rows": len(prepared),
        "codec_frames": codec_frames,
        "text_tokens": total_tokens,
        "split_counts": {"train": split_counts[0], "val": split_counts[1]},
        "codec_storage": storage_mode,
    }


def finalize(
    input_path,
    work_dir,
    tokenizer_path,
    output_path,
    *,
    resume=False,
):
    _require_supported_g2p()
    plan = _load_or_create_plan(input_path, work_dir, resume=True)
    tokenizer_path = Path(tokenizer_path).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"new SentencePiece model does not exist: {tokenizer_path}")
    tokenizer_sha = _hash_file(tokenizer_path)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    output_root = Path(output_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completed = []

    for dataset in plan["datasets"]:
        source_root = Path(dataset["source_root"])
        source_catalog = _read_json(dataset["catalog"])
        target_root = output_root / dataset["relative_root"]
        target_catalog_path = target_root / "catalog.json"
        if target_catalog_path.is_file() and resume:
            target_catalog = _read_json(target_catalog_path)
            if (
                target_catalog.get("tokenizer", {}).get("sha256") != tokenizer_sha
                or target_catalog.get("g2p") != plan["g2p"]
            ):
                raise ValueError(f"existing finalized dataset has different identity: {target_root}")
            completed.append(str(target_root))
            continue
        if target_root.exists():
            raise FileExistsError(
                f"output dataset already exists; choose an empty output or pass --resume "
                f"for a completed matching dataset: {target_root}"
            )

        temporary_root = Path(
            tempfile.mkdtemp(
                dir=output_root,
                prefix=f".{Path(dataset['relative_root']).name}.",
            )
        )
        try:
            output_shards = []
            for source_shard, plan_shard in zip(
                source_catalog["shards"],
                dataset["shards"],
            ):
                prepared_path = _prepared_metadata_path(
                    work_dir,
                    dataset,
                    plan_shard,
                )
                print(f"reindex: {dataset['corpus']} / {source_shard['name']}")
                output_shard = _finalize_shard(
                    source_root,
                    source_shard,
                    prepared_path,
                    temporary_root,
                    tokenizer,
                )
                if output_shard is not None:
                    output_shards.append(output_shard)

            output_catalog = dict(source_catalog)
            output_catalog["shards"] = output_shards
            output_catalog["tokenizer"] = {"sha256": tokenizer_sha}
            output_catalog["g2p"] = plan["g2p"]
            output_catalog["migration"] = {
                "source_catalog_sha256": dataset["catalog_sha256"],
                "tool": "scripts/change_phoneme/change_phoneme.py",
            }
            _write_json(temporary_root / "catalog.json", output_catalog)
            target_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary_root, target_root)
        except BaseException:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise
        completed.append(str(target_root))

    return {
        "datasets": completed,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha,
        "g2p": plan["g2p"],
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="re-phonemize metadata and write the SentencePiece text corpus",
    )
    prepare_parser.add_argument("--input", default="DATASET/compiled")
    prepare_parser.add_argument("--work-dir", default="DATASET/change_phoneme_v1")
    prepare_parser.add_argument(
        "--phoneme-corpus",
        default="DATASET/tokenize/phonemes_v1.txt",
    )
    prepare_parser.add_argument("--batch-size", type=int, default=4096)
    prepare_parser.add_argument("--num-threads", type=int)
    prepare_parser.add_argument("--on-error", choices=("fail", "skip"), default="skip")
    prepare_parser.add_argument("--reject-unk", action="store_true")
    prepare_parser.add_argument("--resume", action="store_true")

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="rebuild text ids and rows with the new SentencePiece model",
    )
    finalize_parser.add_argument("--input", default="DATASET/compiled")
    finalize_parser.add_argument("--work-dir", default="DATASET/change_phoneme_v1")
    finalize_parser.add_argument(
        "--tokenizer",
        default="DATASET/tokenize/models_v1/spm.model",
    )
    finalize_parser.add_argument("--output", default="DATASET/complied_v1")
    finalize_parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(
                args.input,
                args.work_dir,
                args.phoneme_corpus,
                batch_size=args.batch_size,
                num_threads=args.num_threads,
                on_error=args.on_error,
                reject_unk=args.reject_unk,
                resume=args.resume,
            )
        else:
            result = finalize(
                args.input,
                args.work_dir,
                args.tokenizer,
                args.output,
                resume=args.resume,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
