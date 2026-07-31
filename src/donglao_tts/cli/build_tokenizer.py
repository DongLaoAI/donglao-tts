"""Train the SentencePiece tokenizer used by donglao-tts."""

import argparse
import os

import sentencepiece as spm

from donglao_tts.config import load_config


def build_sentencepiece(
    input_path,
    model_prefix,
    vocab_size,
    character_coverage,
    model_type,
    user_defined_symbols=None,
    byte_fallback=False,
):
    parent = os.path.dirname(os.path.abspath(model_prefix))
    os.makedirs(parent, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        byte_fallback=byte_fallback,
        user_defined_symbols=user_defined_symbols or [],
    )
    return f"{model_prefix}.model"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="donglao-tts YAML configuration")
    parser.add_argument("--input", required=True, help="input phoneme corpus text file")
    parser.add_argument(
        "--model-prefix",
        required=True,
        help="output prefix; SentencePiece writes .model and .vocab files",
    )
    args = parser.parse_args()

    tokenizer = load_config(args.config)["tokenizer"]
    model_path = build_sentencepiece(
        args.input,
        args.model_prefix,
        vocab_size=tokenizer.get("vocab_size", 1000),
        character_coverage=tokenizer.get("character_coverage", 1.0),
        model_type=tokenizer.get("model_type", "unigram"),
        user_defined_symbols=tokenizer.get("user_defined_symbols", []),
        byte_fallback=tokenizer.get("byte_fallback", False),
    )
    print(f"wrote {model_path} and {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
