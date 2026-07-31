#!/usr/bin/env python3
"""Stream Emilia English WebDataset shards into donglao-tts compiled shards."""

# ruff: noqa: E402

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PACKAGE = _SOURCE_ROOT / "src"
if _SOURCE_PACKAGE.is_dir():
    sys.path.insert(0, str(_SOURCE_PACKAGE))

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio
import yaml
from donglao_g2p import Pipeline
from huggingface_hub import HfApi, HfFileSystem

from convert_phoaudiobook import (
    _append_chunk,
    _existing_ids,
    _load_json,
    _rewrite_chunk,
    _save_json_atomic,
)
from donglao_tts.data.compiled import (
    _hash_file,
    _validate_entry,
    compile_dataset,
    load_catalog,
)
from donglao_tts.models.codec.moss_codec import MossCodec


DEFAULT_REPO_ID = "amphion/Emilia-Dataset"
DEFAULT_REVISION = "d7f2f7340a6385696f3766c8049fa920a4707c07"
DEFAULT_CORPUS = "emilia-en"
LANGUAGE = "en"
SHARD_PREFIX = "Emilia/EN/"
SHARD_SUFFIX = ".tar"


def _english_shards(repo_info):
    return sorted(
        sibling.rfilename
        for sibling in repo_info.siblings
        if sibling.rfilename.startswith(SHARD_PREFIX)
        and sibling.rfilename.endswith(SHARD_SUFFIX)
    )


def iter_webdataset_records(source, *, max_json_bytes=1024 * 1024, max_audio_bytes=100 * 1024**2):
    """Yield paired Emilia metadata and MP3 bytes from a streaming tar file."""
    pending = {}
    with tarfile.open(fileobj=source, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_path = Path(member.name)
            suffix = member_path.suffix.lower()
            if suffix not in (".json", ".mp3"):
                continue
            size_limit = max_json_bytes if suffix == ".json" else max_audio_bytes
            if member.size > size_limit:
                raise ValueError(
                    f"WebDataset member exceeds safety limit ({member.size} bytes): {member.name}"
                )
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read WebDataset member: {member.name}")
            key = str(member_path.with_suffix(""))
            pair = pending.setdefault(key, {})
            pair[suffix] = extracted.read()
            if ".json" not in pair or ".mp3" not in pair:
                continue
            try:
                metadata = json.loads(pair[".json"])
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid Emilia metadata: {key}.json") from exc
            yield metadata, pair[".mp3"]
            del pending[key]
    if pending:
        preview = ", ".join(sorted(pending)[:3])
        raise ValueError(f"unpaired WebDataset members at end of shard: {preview}")


def _decode_audio(audio_bytes, codec):
    samples, sampling_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[0] != codec.num_channels:
        waveform = waveform.repeat(codec.num_channels, 1)
    if sampling_rate != codec.sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sampling_rate,
            codec.sampling_rate,
        )
    return waveform


def encode_example(metadata, audio_bytes, phoneme, codec):
    for field in ("id", "text", "speaker", "language"):
        if field not in metadata:
            raise ValueError(f"Emilia metadata is missing {field!r}")
    if str(metadata["language"]).lower() != LANGUAGE:
        raise ValueError(
            f"expected English metadata, got language={metadata['language']!r}"
        )
    source_id = str(metadata["id"])
    codes = codec.encode(_decode_audio(audio_bytes, codec))
    return {
        "id": source_id,
        "source_id": source_id,
        "speaker": str(metadata["speaker"]),
        "text": str(metadata["text"]),
        "phoneme": phoneme,
        "codec": codes.cpu().numpy().T.tolist(),
    }


def _validate_output(
    output,
    corpus,
    tokenizer_path,
    num_quantizers,
    codebook_size,
    val_ratio,
    seed,
):
    if not (output / "catalog.json").is_file():
        return
    _, catalog = load_catalog(output)
    expected = (
        catalog["corpus"],
        catalog["language"],
        catalog["tokenizer"]["sha256"],
        catalog["codec"]["num_quantizers"],
        catalog["codec"]["codebook_size"],
        catalog["split"]["val_ratio"],
        catalog["split"]["seed"],
    )
    actual = (
        corpus,
        LANGUAGE,
        _hash_file(tokenizer_path),
        num_quantizers,
        codebook_size,
        val_ratio,
        seed,
    )
    if actual != expected:
        raise ValueError(
            "existing output does not match corpus, language, tokenizer, codec, or split settings"
        )


def _validate_pending_entry(
    entry,
    *,
    manifest_path,
    line_number,
    corpus,
    num_quantizers,
    codebook_size,
    tokenizer,
):
    _validate_entry(
        entry,
        manifest_path,
        line_number,
        corpus,
        num_quantizers,
        codebook_size,
    )
    text_ids = tokenizer.encode(str(entry["phoneme"]), out_type=int)
    if not text_ids:
        raise ValueError(
            f"tokenizer produced no tokens at {manifest_path}:{line_number}"
        )
    if min(text_ids) < 0:
        raise ValueError(
            f"tokenizer produced a negative id at {manifest_path}:{line_number}"
        )


def _read_pending_tolerant(path):
    if not path.is_file():
        return [], []
    entries = []
    errors = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                entry = json.loads(line)
                if not str(entry["source_id"]):
                    raise ValueError("empty source_id")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                errors.append((line_number, exc))
                continue
            entries.append((line_number, entry))
    return entries, errors


def convert_emilia_en(
    *,
    config_path,
    output_path,
    work_dir,
    tokenizer_path=None,
    repo_id=DEFAULT_REPO_ID,
    corpus=DEFAULT_CORPUS,
    revision=DEFAULT_REVISION,
    token=None,
    resume=False,
    append=False,
    strict=False,
    g2p_batch_size=64,
    shard_size=4096,
    val_ratio=0.01,
    seed=42,
    codebook_size=1024,
    max_samples=None,
    max_source_shards=None,
    start_shard=0,
    codec=None,
    pipeline=None,
    repo_info=None,
    filesystem=None,
):
    """Convert Emilia English incrementally without downloading the full raw dataset."""
    if g2p_batch_size < 1 or shard_size < 1:
        raise ValueError("batch and shard sizes must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    if max_source_shards is not None and max_source_shards < 1:
        raise ValueError("max_source_shards must be positive")
    if start_shard < 0:
        raise ValueError("start_shard cannot be negative")

    with open(config_path, "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    tokenizer_path = str(
        Path(tokenizer_path or config["tokenizer"]["model_path"]).resolve()
    )
    num_quantizers = int(config["codec"].get("num_quantizers", 8))
    output = Path(output_path).resolve()
    work_dir = Path(work_dir).resolve()
    state_path = work_dir / "state.json"
    chunk_path = work_dir / "pending.phon.jsonl"

    if output.exists() and not (resume or append):
        raise FileExistsError(
            f"output exists; use --resume for an interrupted import or --append: {output}"
        )
    if resume and not state_path.is_file() and output.exists():
        raise FileNotFoundError(
            f"cannot resume existing output: missing {state_path}"
        )
    _validate_output(
        output,
        corpus,
        tokenizer_path,
        num_quantizers,
        codebook_size,
        val_ratio,
        seed,
    )

    if repo_info is None:
        try:
            repo_info = HfApi().dataset_info(
                repo_id,
                revision=revision,
                token=token,
                files_metadata=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "cannot access Emilia; accept its gated terms and authenticate with "
                "`hf auth login`"
            ) from exc
    resolved_revision = repo_info.sha
    source_shards = _english_shards(repo_info)
    if not source_shards:
        raise RuntimeError(
            f"no English shards matching {SHARD_PREFIX}*{SHARD_SUFFIX}"
        )
    if start_shard >= len(source_shards):
        raise ValueError(
            f"start_shard={start_shard} is outside {len(source_shards)} source shards"
        )

    identity = {
        "repo_id": repo_id,
        "revision": resolved_revision,
        "selection": f"{SHARD_PREFIX}*{SHARD_SUFFIX}",
        "source_shards": len(source_shards),
        "corpus": corpus,
        "language": LANGUAGE,
    }
    if state_path.is_file():
        state = _load_json(state_path)
        for key, value in identity.items():
            if state.get(key) != value:
                raise ValueError(
                    f"resume state {key}={state.get(key)!r} does not match {value!r}"
                )
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        state = {
            **identity,
            "next_shard": start_shard,
            "next_sample": 0,
            "seen": 0,
            "compiled": 0,
            "skipped": 0,
        }
        _save_json_atomic(state_path, state)

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    existing_ids = _existing_ids(output)
    pending = []
    pending_entries, pending_parse_errors = _read_pending_tolerant(chunk_path)
    for line_number, exc in pending_parse_errors:
        state["skipped"] += 1
        print(f"remove malformed pending record at line {line_number}: {exc}")
    for line_number, entry in pending_entries:
        if f"{corpus}:{entry['source_id']}" in existing_ids:
            continue
        try:
            _validate_pending_entry(
                entry,
                manifest_path=chunk_path,
                line_number=line_number,
                corpus=corpus,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            state["skipped"] += 1
            print(f"remove invalid pending record at line {line_number}: {exc}")
            continue
        pending.append(entry)
    _rewrite_chunk(chunk_path, pending)
    pending_ids = {str(entry["source_id"]) for entry in pending}
    _save_json_atomic(state_path, state)

    codec = codec or MossCodec.from_config(config_path)
    pipeline = pipeline or Pipeline()
    filesystem = filesystem or HfFileSystem(token=token)

    def commit_pending():
        nonlocal pending, pending_ids, existing_ids
        if not pending:
            return
        append_mode = (output / "catalog.json").is_file()
        catalog = compile_dataset(
            [(chunk_path, corpus, LANGUAGE)],
            output,
            tokenizer_path,
            shard_size=shard_size,
            val_ratio=val_ratio,
            seed=seed,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            append=append_mode,
            check_existing_ids=False,
        )
        for entry in pending:
            existing_ids.add(f"{corpus}:{entry['source_id']}")
        state["compiled"] += len(pending)
        state["total_shards"] = len(catalog["shards"])
        pending = []
        pending_ids = set()
        _rewrite_chunk(chunk_path, pending)
        _save_json_atomic(state_path, state)
        print(
            f"committed: {state['compiled']} utterances, "
            f"{state['skipped']} skipped, {state['total_shards']} compiled shards"
        )

    processed_this_run = 0
    source_shards_this_run = 0
    stop_requested = False

    def process_batch(examples):
        nonlocal pending
        texts = [str(metadata["text"]) for _, metadata, _ in examples]
        try:
            phonemes = pipeline.phonemize_batch(texts)
        except Exception as exc:
            if strict:
                raise RuntimeError("Emilia English phonemization failed") from exc
            state["skipped"] += len(examples)
            print(f"skip G2P batch of {len(examples)} examples: {exc}")
            return
        if len(phonemes) != len(examples):
            raise RuntimeError(
                f"phonemizer returned {len(phonemes)} results for {len(examples)} texts"
            )

        new_entries = []
        for (sample_index, metadata, audio_bytes), phoneme in zip(examples, phonemes):
            source_id = str(metadata.get("id", ""))
            utterance_id = f"{corpus}:{source_id}"
            if utterance_id in existing_ids or source_id in pending_ids:
                continue
            try:
                if not isinstance(phoneme, str) or not phoneme.strip():
                    raise ValueError("phonemizer returned an empty phoneme")
                entry = encode_example(metadata, audio_bytes, phoneme, codec)
                _validate_pending_entry(
                    entry,
                    manifest_path=chunk_path,
                    line_number=len(pending) + len(new_entries) + 1,
                    corpus=corpus,
                    num_quantizers=num_quantizers,
                    codebook_size=codebook_size,
                    tokenizer=tokenizer,
                )
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"failed to encode Emilia sample at source index {sample_index}"
                    ) from exc
                state["skipped"] += 1
                print(f"skip Emilia sample {source_id or sample_index}: {exc}")
                continue
            new_entries.append(entry)
            pending_ids.add(source_id)
        if new_entries:
            _append_chunk(chunk_path, new_entries)
            pending.extend(new_entries)
        if len(pending) >= shard_size:
            commit_pending()

    first_shard = int(state["next_shard"])
    final_shard = len(source_shards)
    if max_source_shards is not None:
        final_shard = min(final_shard, first_shard + max_source_shards)

    for shard_index in range(first_shard, final_shard):
        shard_name = source_shards[shard_index]
        resume_sample = int(state["next_sample"]) if shard_index == first_shard else 0
        hub_path = f"datasets/{repo_id}@{resolved_revision}/{shard_name}"
        print(
            f"source shard {shard_index + 1}/{len(source_shards)}: "
            f"{shard_name} (resume sample {resume_sample})"
        )
        batch = []
        with filesystem.open(hub_path, "rb") as source:
            for sample_index, (metadata, audio_bytes) in enumerate(
                iter_webdataset_records(source)
            ):
                if sample_index < resume_sample:
                    continue
                if max_samples is not None and processed_this_run >= max_samples:
                    stop_requested = True
                    break

                source_id = str(metadata.get("id", ""))
                utterance_id = f"{corpus}:{source_id}"
                if utterance_id not in existing_ids and source_id not in pending_ids:
                    batch.append((sample_index, metadata, audio_bytes))
                processed_this_run += 1
                state["seen"] += 1
                state["next_shard"] = shard_index
                state["next_sample"] = sample_index + 1
                if len(batch) == g2p_batch_size:
                    process_batch(batch)
                    batch = []
                    _save_json_atomic(state_path, state)

        if batch:
            process_batch(batch)
        if stop_requested:
            commit_pending()
            _save_json_atomic(state_path, state)
            break

        state["next_shard"] = shard_index + 1
        state["next_sample"] = 0
        source_shards_this_run += 1
        commit_pending()
        _save_json_atomic(state_path, state)

    state["source_shards_processed_this_run"] = source_shards_this_run
    commit_pending()
    _save_json_atomic(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="donglao-tts YAML config")
    parser.add_argument("--output", required=True, help="compiled Emilia English directory")
    parser.add_argument(
        "--work-dir",
        required=True,
        help="persistent resume state and pending compiled-shard manifest",
    )
    parser.add_argument("--tokenizer", help="defaults to tokenizer.model_path in config")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--token",
        help="Hugging Face token; prefer `hf auth login` to avoid shell history exposure",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--g2p-batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument(
        "--max-source-shards",
        type=int,
        help="process at most this many source tar shards in the current run",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="process at most this many source rows in the current run",
    )
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help="confirm you accepted Emilia's gated terms and CC BY-NC 4.0 license",
    )
    args = parser.parse_args()
    if not args.accept_terms:
        parser.error(
            "--accept-terms is required; accept the dataset terms at "
            "https://huggingface.co/datasets/amphion/Emilia-Dataset first"
        )
    if args.resume and args.append:
        parser.error("choose --resume or --append, not both")

    state = convert_emilia_en(
        config_path=args.config,
        output_path=args.output,
        work_dir=args.work_dir,
        tokenizer_path=args.tokenizer,
        repo_id=args.repo_id,
        corpus=args.corpus,
        revision=args.revision,
        token=args.token,
        resume=args.resume,
        append=args.append,
        strict=args.strict,
        g2p_batch_size=args.g2p_batch_size,
        shard_size=args.shard_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        codebook_size=args.codebook_size,
        max_samples=args.max_samples,
        max_source_shards=args.max_source_shards,
        start_shard=args.start_shard,
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
