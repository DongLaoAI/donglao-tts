"""Add phonemes to a donglao-tts JSONL manifest."""

import argparse
import json
import os

from donglao_g2p import Pipeline

from donglao_tts.cli._io import atomic_text_writer


def phonemize_manifest(input_path, output_path, pipeline):
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("input and output manifest paths must be different")

    entries = []
    with open(input_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                entry = json.loads(line)
                entry["text"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid manifest entry at line {line_number}") from exc
            entries.append(entry)

    texts = [entry["text"] for entry in entries]
    print(f"phonemizing {len(texts)} lines from {input_path}")
    phonemes = pipeline.phonemize_batch(texts)
    if len(phonemes) != len(entries):
        raise RuntimeError(
            f"phonemizer returned {len(phonemes)} results for {len(entries)} manifest entries"
        )

    with atomic_text_writer(output_path) as destination:
        for entry, phoneme in zip(entries, phonemes):
            entry["phoneme"] = phoneme
            destination.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input JSONL manifest")
    parser.add_argument("--output", required=True, help="output phonemized JSONL manifest")
    parser.add_argument(
        "--lang",
        default="vi",
        help="deprecated compatibility option; donglao-g2p routing is automatic",
    )
    args = parser.parse_args()

    count = phonemize_manifest(args.input, args.output, Pipeline())
    print(f"wrote {count} entries to {args.output}")


if __name__ == "__main__":
    main()
