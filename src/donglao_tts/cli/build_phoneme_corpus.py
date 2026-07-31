"""Build a SentencePiece training corpus from one or more text manifests."""

import argparse
import json

from donglao_g2p import Pipeline

from donglao_tts.cli._io import atomic_text_writer


def _load_texts(manifest_path):
    texts = []
    with open(manifest_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"invalid text manifest entry at {manifest_path}:{line_number}"
                ) from exc
    return texts


def build_phoneme_corpus(manifest_languages, output_path):
    pipeline = Pipeline()
    written = 0
    with atomic_text_writer(output_path) as destination:
        for manifest_path, language in manifest_languages:
            texts = _load_texts(manifest_path)
            print(
                f"phonemizing {len(texts)} lines from {manifest_path} "
                f"(label={language}, automatic language routing)"
            )
            phonemes = pipeline.phonemize_batch(texts)
            if len(phonemes) != len(texts):
                raise RuntimeError(
                    f"phonemizer returned {len(phonemes)} results for {len(texts)} texts"
                )
            for phoneme in phonemes:
                destination.write(phoneme + "\n")
                written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        nargs=2,
        action="append",
        metavar=("PATH", "LANG"),
        required=True,
        help="input text manifest and legacy language label; routing is automatic",
    )
    parser.add_argument("--output", required=True, help="output phoneme corpus text file")
    args = parser.parse_args()

    count = build_phoneme_corpus(args.manifest, args.output)
    print(f"wrote {count} phoneme lines to {args.output}")


if __name__ == "__main__":
    main()
