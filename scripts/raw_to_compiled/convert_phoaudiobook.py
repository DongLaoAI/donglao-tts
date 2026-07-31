#!/usr/bin/env python3
"""Stream thivux/phoaudiobook from Hugging Face into donglao-tts compiled shards."""

# ruff: noqa: E402

import argparse
import io
import json
import os
import sys
from pathlib import Path


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PACKAGE = _SOURCE_ROOT / "src"
if _SOURCE_PACKAGE.is_dir():
    sys.path.insert(0, str(_SOURCE_PACKAGE))

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from donglao_g2p import Pipeline
from huggingface_hub import HfApi

from donglao_tts.cli._io import atomic_text_writer
from donglao_tts.data.compiled import (
    _hash_file,
    _load_existing_ids,
    compile_dataset,
    load_catalog,
)
from donglao_tts.models.codec.moss_codec import MossCodec


DEFAULT_REPO_ID = "thivux/phoaudiobook"
DEFAULT_CORPUS = "phoaudiobook"


def _save_json_atomic(path, value):
    with atomic_text_writer(path) as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _audio_bytes(audio):
    if not isinstance(audio, dict):
        raise TypeError(
            "expected an undecoded Hugging Face Audio dictionary; "
            "ensure the streaming dataset uses decode(False)"
        )
    payload = audio.get("bytes")
    if payload is not None:
        return io.BytesIO(payload)
    path = audio.get("path")
    if path:
        return path
    raise ValueError("audio example contains neither bytes nor path")


def decode_audio_for_codec(audio, codec):
    """Decode an HF audio example without requiring TorchCodec."""
    samples, sampling_rate = sf.read(
        _audio_bytes(audio), dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[0] != codec.num_channels:
        waveform = waveform.repeat(codec.num_channels, 1)
    if sampling_rate != codec.sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform, sampling_rate, codec.sampling_rate
        )
    return waveform


def encode_example(example, source_id, phoneme, codec):
    for field in ("audio", "text", "speaker"):
        if field not in example:
            raise ValueError(f"PhoAudiobook example is missing {field!r}")
    codes = codec.encode(decode_audio_for_codec(example["audio"], codec))
    return {
        "id": source_id,
        "source_id": source_id,
        "speaker": str(example["speaker"]),
        "text": str(example["text"]),
        "phoneme": phoneme,
        "codec": codes.cpu().numpy().T.tolist(),
    }


def _read_chunk(path):
    entries = []
    if not path.is_file():
        return entries
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                entry = json.loads(line)
                entry["source_id"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid work chunk at {path}:{line_number}") from exc
            entries.append(entry)
    return entries


def _rewrite_chunk(path, entries):
    with atomic_text_writer(path) as destination:
        for entry in entries:
            destination.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _append_chunk(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        for entry in entries:
            destination.write(json.dumps(entry, ensure_ascii=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def _existing_ids(output):
    output = Path(output)
    if not (output / "catalog.json").is_file():
        return set()
    root, catalog = load_catalog(output)
    return _load_existing_ids(root, catalog)


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
        "vi",
        _hash_file(tokenizer_path),
        num_quantizers,
        codebook_size,
        val_ratio,
        seed,
    )
    if actual != expected:
        raise ValueError(
            "existing output does not match corpus, tokenizer, codec, or split settings"
        )


def _load_stream(repo_id, split, revision, token, cache_dir, start_index):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'install the importer dependencies with: pip install "datasets>=4,<6" soundfile'
        ) from exc
    try:
        dataset = load_dataset(
            repo_id,
            split=split,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            streaming=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to open gated PhoAudiobook; accept its terms in the browser and run "
            "`hf auth login` before retrying"
        ) from exc
    if hasattr(dataset, "decode"):
        dataset = dataset.decode(False)
    else:
        from datasets import Audio

        dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset.skip(start_index) if start_index else dataset


def convert_phoaudiobook(
    *,
    config_path,
    output_path,
    work_dir,
    tokenizer_path=None,
    repo_id=DEFAULT_REPO_ID,
    corpus=DEFAULT_CORPUS,
    split="train",
    revision="main",
    token=None,
    cache_dir=None,
    resume=False,
    append=False,
    strict=False,
    g2p_batch_size=64,
    shard_size=4096,
    val_ratio=0.01,
    seed=42,
    codebook_size=1024,
    max_samples=None,
    start_index=0,
    codec=None,
    pipeline=None,
    dataset=None,
    resolved_revision=None,
):
    """Convert a streaming PhoAudiobook split, committing one compiled shard at a time."""
    if g2p_batch_size < 1 or shard_size < 1:
        raise ValueError("batch and shard sizes must be positive")
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
    if resume and not state_path.is_file():
        raise FileNotFoundError(f"cannot resume: missing {state_path}")
    _validate_output(
        output,
        corpus,
        tokenizer_path,
        num_quantizers,
        codebook_size,
        val_ratio,
        seed,
    )

    if resolved_revision is None:
        try:
            resolved_revision = HfApi().dataset_info(
                repo_id, revision=revision, token=token
            ).sha
        except Exception as exc:
            raise RuntimeError(
                "cannot access PhoAudiobook metadata; accept the gated dataset terms and "
                "authenticate with `hf auth login`"
            ) from exc

    identity = {
        "repo_id": repo_id,
        "revision": resolved_revision,
        "split": split,
        "corpus": corpus,
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
            "next_index": start_index,
            "seen": 0,
            "compiled": 0,
            "skipped": 0,
        }
        _save_json_atomic(state_path, state)

    existing_ids = _existing_ids(output)
    pending = [
        entry
        for entry in _read_chunk(chunk_path)
        if f"{corpus}:{entry['source_id']}" not in existing_ids
    ]
    _rewrite_chunk(chunk_path, pending)
    pending_ids = {str(entry["source_id"]) for entry in pending}

    codec = codec or MossCodec.from_config(config_path)
    pipeline = pipeline or Pipeline()
    if dataset is None:
        dataset = _load_stream(
            repo_id,
            split,
            resolved_revision,
            token,
            cache_dir,
            int(state["next_index"]),
        )

    def commit_pending():
        nonlocal pending, pending_ids, existing_ids
        if not pending:
            return
        append_mode = (output / "catalog.json").is_file()
        catalog = compile_dataset(
            [(chunk_path, corpus, "vi")],
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
            f"{state['skipped']} skipped, {state['total_shards']} shards"
        )

    absolute_index = int(state["next_index"])
    processed_this_run = 0
    batch = []

    def process_batch(examples):
        nonlocal pending
        texts = [str(example["text"]) for _, example in examples]
        try:
            phonemes = pipeline.phonemize_batch(texts)
        except Exception as exc:
            if strict:
                raise RuntimeError("PhoAudiobook phonemization failed") from exc
            state["skipped"] += len(examples)
            print(f"skip G2P batch of {len(examples)} examples: {exc}")
            return
        if len(phonemes) != len(examples):
            raise RuntimeError(
                f"phonemizer returned {len(phonemes)} results for {len(examples)} texts"
            )

        new_entries = []
        for (index, example), phoneme in zip(examples, phonemes):
            source_id = f"{split}:{index:09d}"
            utterance_id = f"{corpus}:{source_id}"
            if utterance_id in existing_ids or source_id in pending_ids:
                continue
            try:
                entry = encode_example(example, source_id, phoneme, codec)
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"failed to encode PhoAudiobook {source_id}"
                    ) from exc
                state["skipped"] += 1
                print(f"skip {source_id}: {exc}")
                continue
            new_entries.append(entry)
            pending_ids.add(source_id)
        if new_entries:
            _append_chunk(chunk_path, new_entries)
            pending.extend(new_entries)
        if len(pending) >= shard_size:
            commit_pending()

    for example in dataset:
        if max_samples is not None and processed_this_run >= max_samples:
            break
        source_id = f"{split}:{absolute_index:09d}"
        utterance_id = f"{corpus}:{source_id}"
        if utterance_id not in existing_ids and source_id not in pending_ids:
            batch.append((absolute_index, example))
        absolute_index += 1
        processed_this_run += 1
        state["seen"] += 1
        state["next_index"] = absolute_index
        if len(batch) == g2p_batch_size:
            process_batch(batch)
            batch = []
            _save_json_atomic(state_path, state)
    if batch:
        process_batch(batch)
    commit_pending()
    _save_json_atomic(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="donglao-tts YAML config")
    parser.add_argument("--output", required=True, help="compiled PhoAudiobook directory")
    parser.add_argument(
        "--work-dir",
        required=True,
        help="persistent resume state and pending shard directory",
    )
    parser.add_argument("--tokenizer", help="defaults to tokenizer.model_path in config")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--token",
        help="Hugging Face token; prefer `hf auth login` to avoid shell history exposure",
    )
    parser.add_argument("--cache-dir", help="Hugging Face streaming cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--g2p-batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="stop after this many source rows in the current run (useful for smoke tests)",
    )
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help="confirm you accepted PhoAudiobook's gated research/education terms",
    )
    args = parser.parse_args()
    if not args.accept_terms:
        parser.error(
            "--accept-terms is required; accept the dataset terms at "
            "https://huggingface.co/datasets/thivux/phoaudiobook first"
        )
    if args.resume and args.append:
        parser.error("choose --resume or --append, not both")

    state = convert_phoaudiobook(
        config_path=args.config,
        output_path=args.output,
        work_dir=args.work_dir,
        tokenizer_path=args.tokenizer,
        repo_id=args.repo_id,
        corpus=args.corpus,
        split=args.split,
        revision=args.revision,
        token=args.token,
        cache_dir=args.cache_dir,
        resume=args.resume,
        append=args.append,
        strict=args.strict,
        g2p_batch_size=args.g2p_batch_size,
        shard_size=args.shard_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        codebook_size=args.codebook_size,
        max_samples=args.max_samples,
        start_index=args.start_index,
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
