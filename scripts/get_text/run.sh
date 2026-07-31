#!/usr/bin/env bash
python3 scripts/get_text/export_text.py \
  --input DATASET/compiled \
  --output DATASET/text/language_text.csv \
  --on-error skip \
  --force
