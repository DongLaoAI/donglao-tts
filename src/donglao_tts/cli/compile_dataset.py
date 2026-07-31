"""Compile phonemized JSONL manifests into memory-mapped training shards."""

import argparse

from donglao_tts.data.compiled import compile_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        nargs=3,
        action="append",
        metavar=("PATH", "CORPUS", "LANG"),
        required=True,
        help="phonemized manifest, stable corpus name, and language; repeat as needed",
    )
    parser.add_argument("--tokenizer", required=True, help="SentencePiece model")
    parser.add_argument(
        "--output",
        required=True,
        help="compiled directory dedicated to the selected corpus",
    )
    parser.add_argument("--shard-size", type=int, default=4096, help="utterances per shard")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-quantizers", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument(
        "--append",
        action="store_true",
        help="add immutable shards to an existing compatible compiled dataset",
    )
    args = parser.parse_args()

    catalog = compile_dataset(
        args.manifest,
        args.output,
        args.tokenizer,
        shard_size=args.shard_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size,
        append=args.append,
    )
    rows = sum(shard["rows"] for shard in catalog["shards"])
    frames = sum(shard["codec_frames"] for shard in catalog["shards"])
    print(
        f"compiled dataset at {args.output}: "
        f"{rows} utterances, {frames} codec frames, {len(catalog['shards'])} shards"
    )


if __name__ == "__main__":
    main()
