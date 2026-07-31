#!/usr/bin/env bash
python3 scripts/change_phoneme/change_phoneme.py finalize \
  --input DATASET/compiled \
  --work-dir DATASET/change_phoneme_v1 \
  --tokenizer DATASET/tokenize/models_v1/spm.model \
  --output DATASET/complied_v1 \
  --resume
