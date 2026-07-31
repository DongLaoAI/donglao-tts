#!/usr/bin/env bash
python3 -m donglao_tts.cli.build_tokenizer \
  --config configs/base.yaml \
  --input DATASET/tokenize/phonemes_v1.txt \
  --model-prefix DATASET/tokenize/models_v1/spm
