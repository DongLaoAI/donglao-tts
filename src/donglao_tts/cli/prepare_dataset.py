"""Encode an audio metadata table into a donglao-tts RVQ manifest."""

import argparse
import csv
import json
import os

from donglao_tts.cli._io import atomic_text_writer
from donglao_tts.models.codec.moss_codec import MossCodec

_REQUIRED_COLUMNS = {"audio_path", "speaker_id", "text"}


def prepare_dataset(metadata_path, output_path, codec, audio_root=".", strict=False):
    audio_root = os.path.abspath(audio_root)
    written = 0
    skipped = 0

    with open(metadata_path, "r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="|")
        missing = _REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"metadata is missing required columns: {', '.join(sorted(missing))}")

        with atomic_text_writer(output_path) as destination:
            for line_number, row in enumerate(reader, start=2):
                audio_path = row["audio_path"]
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(audio_root, audio_path)
                try:
                    codes = codec.encode_file(audio_path)
                except Exception as exc:
                    if strict:
                        raise RuntimeError(
                            f"failed to encode metadata line {line_number}: {row['audio_path']}"
                        ) from exc
                    skipped += 1
                    print(f"skip line {line_number} ({row['audio_path']}): {exc}")
                    continue

                entry = {
                    "id": written,
                    # Keep a stable source identity even if unreadable rows are later added,
                    # removed, or reordered. The sequential id remains for compatibility.
                    "source_id": row["audio_path"],
                    "speaker": row["speaker_id"],
                    "text": row["text"],
                    "codec": codes.cpu().numpy().T.tolist(),
                }
                destination.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
                if written % 500 == 0:
                    print(f"processed {written} entries")

    return written, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="donglao-tts YAML configuration")
    parser.add_argument("--metadata", required=True, help="pipe-delimited metadata file")
    parser.add_argument("--output", required=True, help="output JSONL manifest")
    parser.add_argument(
        "--audio-root",
        default=".",
        help="base directory for relative audio_path values (default: current directory)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop at the first unreadable audio file instead of skipping it",
    )
    args = parser.parse_args()

    codec = MossCodec.from_config(args.config)
    written, skipped = prepare_dataset(
        args.metadata,
        args.output,
        codec,
        audio_root=args.audio_root,
        strict=args.strict,
    )
    print(f"wrote {written} entries to {args.output} ({skipped} skipped)")


if __name__ == "__main__":
    main()
