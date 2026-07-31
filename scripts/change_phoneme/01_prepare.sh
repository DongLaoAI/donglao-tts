#!/usr/bin/env bash
python3 scripts/change_phoneme/change_phoneme.py prepare \
  --input DATASET/compiled \
  --work-dir DATASET/change_phoneme_v1 \
  --phoneme-corpus DATASET/tokenize/phonemes_v1.txt \
  --batch-size 4096 \
  --on-error skip \
  --reject-unk \
  --resume
