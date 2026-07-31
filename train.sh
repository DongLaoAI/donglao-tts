#!/bin/bash
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES=1 python3 -m src.donglao_tts.cli.train --config configs/base.yaml
