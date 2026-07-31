#!/usr/bin/env python3
"""End-to-end raw audio metadata -> donglao-tts compiled dataset.

This is intentionally a single-command orchestration script. It can run from a source checkout
or on a machine where ``donglao-tts`` is installed.

Expected pipe-delimited metadata:

    audio_path|speaker_id|text
    audio/0001.wav|speaker-01|Xin chào.

An optional ``source_id`` column may be supplied. Otherwise the metadata's ``audio_path`` is the
stable source identity used for deduplication and incremental appends.
"""

# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

from donglao_g2p import Pipeline


# Make direct ``python scripts/raw_to_compiled.py`` work from any current working directory when
# the script is still inside a donglao-tts source checkout. On another machine, installing the
# package is sufficient and this fallback is simply ignored.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PACKAGE = _SOURCE_ROOT / "src"
if _SOURCE_PACKAGE.is_dir():
    sys.path.insert(0, str(_SOURCE_PACKAGE))

import yaml

from donglao_tts.data.compiled import _hash_file, compile_dataset, load_catalog
from donglao_tts.models.codec.moss_codec import MossCodec


_REQUIRED_COLUMNS = {"audio_path", "speaker_id", "text"}


def _load_staged_entries(path):
    entries = {}
    if not Path(path).is_file():
        return entries
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                entry = json.loads(line)
                source_id = str(entry["source_id"])
                speaker = str(entry["speaker"])
                text = str(entry["text"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid staging entry at {path}:{line_number}") from exc
            if source_id in entries:
                raise ValueError(f"duplicate source_id in staging manifest: {source_id}")
            entries[source_id] = (speaker, text)
    return entries


def _load_compiled_utterance_ids(output_path):
    output = Path(output_path)
    if not (output / "catalog.json").is_file():
        return set()
    root, catalog = load_catalog(output)
    utterance_ids = set()
    for shard in catalog["shards"]:
        with (root / shard["metadata"]).open("r", encoding="utf-8") as source:
            for line in source:
                utterance_ids.add(json.loads(line)["utterance_id"])
    return utterance_ids


def _flush_and_sync(stream):
    stream.flush()
    os.fsync(stream.fileno())


def build_staging_manifest(
    metadata_path,
    staging_path,
    codec,
    pipeline,
    *,
    audio_root=".",
    g2p_batch_size=64,
    strict=False,
    resume=False,
    compiled_utterance_ids=None,
    corpus=None,
):
    """Encode raw metadata into a resumable phonemized JSONL staging manifest."""
    if g2p_batch_size < 1:
        raise ValueError("g2p_batch_size must be positive")
    staging_path = Path(staging_path)
    if staging_path.exists() and not resume:
        raise FileExistsError(
            f"staging manifest exists; pass --resume-staging or choose another path: "
            f"{staging_path}"
        )
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    staged = _load_staged_entries(staging_path) if resume else {}
    compiled_utterance_ids = compiled_utterance_ids or set()
    audio_root = Path(audio_root).resolve()
    seen_metadata_ids = set()
    written = 0
    skipped = 0
    already_done = 0

    def process_batch(rows, destination):
        nonlocal written, skipped
        texts = [row["text"] for _, row, _, _ in rows]
        try:
            phonemes = pipeline.phonemize_batch(texts)
        except Exception as exc:
            if strict:
                raise RuntimeError("phonemizer failed") from exc
            for line_number, row, _, _ in rows:
                print(f"skip line {line_number} ({row['audio_path']}): phonemizer: {exc}")
            skipped += len(rows)
            return
        if len(phonemes) != len(rows):
            raise RuntimeError(
                f"phonemizer returned {len(phonemes)} results for {len(rows)} texts"
            )

        for (line_number, row, source_id, audio_path), phoneme in zip(rows, phonemes):
            try:
                codes = codec.encode_file(str(audio_path))
                entry = {
                    "id": source_id,
                    "source_id": source_id,
                    "speaker": row["speaker_id"],
                    "text": row["text"],
                    "phoneme": phoneme,
                    "codec": codes.cpu().numpy().T.tolist(),
                }
                destination.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"failed to encode metadata line {line_number}: {row['audio_path']}"
                    ) from exc
                skipped += 1
                print(f"skip line {line_number} ({row['audio_path']}): codec: {exc}")
        _flush_and_sync(destination)
        print(
            f"staging progress: {written} new, {already_done} existing, {skipped} skipped"
        )

    mode = "a" if resume else "x"
    with open(metadata_path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="|")
        missing = _REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"metadata is missing required columns: {', '.join(sorted(missing))}"
            )
        with staging_path.open(mode, encoding="utf-8") as destination:
            batch = []
            for line_number, row in enumerate(reader, start=2):
                source_id = str(row.get("source_id") or row["audio_path"])
                if not source_id:
                    raise ValueError(f"empty source_id at metadata line {line_number}")
                if source_id in seen_metadata_ids:
                    raise ValueError(f"duplicate source_id in metadata: {source_id}")
                seen_metadata_ids.add(source_id)

                staged_value = staged.get(source_id)
                if staged_value is not None:
                    if staged_value != (str(row["speaker_id"]), str(row["text"])):
                        raise ValueError(
                            f"metadata changed for staged source_id {source_id!r}"
                        )
                    already_done += 1
                    continue
                utterance_id = f"{corpus}:{source_id}" if corpus else None
                if utterance_id and utterance_id in compiled_utterance_ids:
                    already_done += 1
                    continue

                audio_path = Path(row["audio_path"])
                if not audio_path.is_absolute():
                    audio_path = audio_root / audio_path
                batch.append((line_number, row, source_id, audio_path))
                if len(batch) == g2p_batch_size:
                    process_batch(batch, destination)
                    batch = []
            if batch:
                process_batch(batch, destination)

    return {
        "written": written,
        "skipped": skipped,
        "already_done": already_done,
        "staging_path": str(staging_path),
    }


def _filter_uncompiled_staging(
    staging_path, output_path, corpus, destination_path, append
):
    existing = _load_compiled_utterance_ids(output_path) if append else set()
    selected = 0
    with open(staging_path, "r", encoding="utf-8") as source:
        with open(destination_path, "w", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, start=1):
                try:
                    entry = json.loads(line)
                    utterance_id = f"{corpus}:{entry['source_id']}"
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"invalid staging entry at {staging_path}:{line_number}"
                    ) from exc
                if utterance_id not in existing:
                    destination.write(line)
                    selected += 1
    return selected


def raw_to_compiled(
    *,
    config_path,
    metadata_path,
    output_path,
    corpus,
    language,
    audio_root=".",
    tokenizer_path=None,
    staging_path=None,
    resume_staging=False,
    append=False,
    strict=False,
    g2p_batch_size=64,
    shard_size=4096,
    val_ratio=0.01,
    seed=42,
    codebook_size=1024,
    codec=None,
    pipeline=None,
):
    """Run the complete raw-to-compiled pipeline and return the final catalog."""
    with open(config_path, "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    tokenizer_path = tokenizer_path or config["tokenizer"]["model_path"]
    tokenizer_path = str(Path(tokenizer_path).resolve())
    num_quantizers = int(config["codec"].get("num_quantizers", 8))

    output = Path(output_path).resolve()
    if append:
        if not (output / "catalog.json").is_file():
            raise FileNotFoundError(f"cannot append: missing {output / 'catalog.json'}")
        _, existing_catalog = load_catalog(output)
        expected = (
            existing_catalog["corpus"],
            existing_catalog["language"],
            existing_catalog["tokenizer"]["sha256"],
            existing_catalog["codec"]["num_quantizers"],
            existing_catalog["codec"]["codebook_size"],
            existing_catalog["split"]["val_ratio"],
            existing_catalog["split"]["seed"],
        )
        actual = (
            corpus,
            language,
            _hash_file(tokenizer_path),
            num_quantizers,
            codebook_size,
            val_ratio,
            seed,
        )
        if actual != expected:
            raise ValueError(
                "append settings do not match corpus, language, tokenizer, codec, or split "
                "configuration in the existing catalog"
            )
    elif output.exists():
        raise FileExistsError(f"refusing to overwrite compiled dataset: {output}")
    if staging_path:
        resolved_staging = Path(staging_path).resolve()
        if resolved_staging == output or output in resolved_staging.parents:
            raise ValueError("--staging-manifest must be outside the compiled --output directory")

    codec = codec or MossCodec.from_config(config_path)
    pipeline = pipeline or Pipeline()
    compiled_ids = _load_compiled_utterance_ids(output) if append else set()

    with tempfile.TemporaryDirectory(prefix="donglao-raw-to-compiled-") as temporary:
        temporary = Path(temporary)
        actual_staging = Path(staging_path).resolve() if staging_path else (
            temporary / f"{corpus}.phon.jsonl"
        )
        stats = build_staging_manifest(
            metadata_path,
            actual_staging,
            codec,
            pipeline,
            audio_root=audio_root,
            g2p_batch_size=g2p_batch_size,
            strict=strict,
            resume=resume_staging,
            compiled_utterance_ids=compiled_ids,
            corpus=corpus,
        )
        compile_input = temporary / "uncompiled.phon.jsonl"
        selected = _filter_uncompiled_staging(
            actual_staging, output, corpus, compile_input, append
        )
        if selected == 0:
            print(
                f"nothing new to compile ({stats['already_done']} already present, "
                f"{stats['skipped']} skipped)"
            )
            if append:
                return load_catalog(output)[1]
            raise RuntimeError("no valid utterances were produced from the raw metadata")

        catalog = compile_dataset(
            [(compile_input, corpus, language)],
            output,
            tokenizer_path,
            shard_size=shard_size,
            val_ratio=val_ratio,
            seed=seed,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            append=append,
        )
        print(
            f"completed {corpus}: {selected} new utterances compiled "
            f"({stats['skipped']} skipped), {len(catalog['shards'])} total shards"
        )
        if staging_path:
            print(f"staging manifest retained at {actual_staging}")
        return catalog


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw TTS audio metadata directly into compiled training shards."
    )
    parser.add_argument("--config", required=True, help="donglao-tts YAML config")
    parser.add_argument("--metadata", required=True, help="pipe-delimited raw metadata")
    parser.add_argument("--audio-root", default=".", help="base for relative audio paths")
    parser.add_argument("--corpus", required=True, help="stable corpus name")
    parser.add_argument(
        "--lang",
        required=True,
        help="compiled dataset language label; donglao-g2p routing is automatic",
    )
    parser.add_argument("--output", required=True, help="compiled directory for this corpus")
    parser.add_argument(
        "--tokenizer",
        help="SentencePiece model; defaults to tokenizer.model_path in --config",
    )
    parser.add_argument(
        "--staging-manifest",
        help="persistent phonemized staging JSONL; enables recovery with --resume-staging",
    )
    parser.add_argument(
        "--resume-staging",
        action="store_true",
        help="continue an existing --staging-manifest without re-encoding completed audio",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append new immutable shards to the corpus output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop at the first G2P/audio error instead of recording a skip",
    )
    parser.add_argument("--g2p-batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codebook-size", type=int, default=1024)
    args = parser.parse_args()
    if args.resume_staging and not args.staging_manifest:
        parser.error("--resume-staging requires --staging-manifest")

    raw_to_compiled(
        config_path=args.config,
        metadata_path=args.metadata,
        output_path=args.output,
        corpus=args.corpus,
        language=args.lang,
        audio_root=args.audio_root,
        tokenizer_path=args.tokenizer,
        staging_path=args.staging_manifest,
        resume_staging=args.resume_staging,
        append=args.append,
        strict=args.strict,
        g2p_batch_size=args.g2p_batch_size,
        shard_size=args.shard_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        codebook_size=args.codebook_size,
    )


if __name__ == "__main__":
    main()
