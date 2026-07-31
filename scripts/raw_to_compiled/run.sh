CUDA_VISIBLE_DEVICES=1 python3 scripts/raw_to_compiled/convert_phoaudiobook.py \
  --accept-terms \
  --config configs/base.yaml \
  --output DATASET/compiled/phoaudiobook \
  --work-dir DATASET/work/phoaudiobook \
  --split train \
  --resume
#   --max-samples 1000 \
