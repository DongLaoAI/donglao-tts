#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/raw_to_compiled/convert_emilia_en.py \
  --accept-terms \
  --config configs/base.yaml \
  --output DATASET/compiled/emilia_en \
  --work-dir DATASET/work/emilia_en \
  --max-samples 500000 \
  --resume
