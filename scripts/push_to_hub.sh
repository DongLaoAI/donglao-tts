#!/bin/sh
set -e

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
python3 -c 'from donglao_tts.hub import _push_to_hub_cli; _push_to_hub_cli()' \
  --config configs/base.yaml \
  --checkpoint checkpoints/run_02/step_90000.pt \
  --repo-id DongLao/DongLao-TTS \
  --out-dir checkpoints/run_02/hub-bundle \
  "$@"
