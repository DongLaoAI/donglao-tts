#!/bin/sh
set -e

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$repo_dir"

PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=1 \
python3 -m donglao_tts.cli.train --config configs/base.yaml "$@"
