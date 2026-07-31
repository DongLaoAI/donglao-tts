# PhoAudiobook to donglao-tts

This importer streams the gated `thivux/phoaudiobook` dataset and commits immutable
donglao-tts shards incrementally. It does not download the full ~400 GB decoded dataset before
processing.

## Access and installation

1. Accept the terms at <https://huggingface.co/datasets/thivux/phoaudiobook>.
2. Authenticate with `hf auth login`.
3. Install the project and importer dependencies:

```bash
python -m pip install -e .
python -m pip install -r scripts/raw_to_compiled/requirements.txt
```

Confirm that authentication can see the gated repository:

```bash
hf auth whoami
```

Do not distribute the raw data or compiled artifacts. The dataset terms restrict use to
research or education, prohibit redistribution, and require citation of the ACL 2025 paper.

Run commands from the `donglao_tts` repository root. Verify these config values first:

```yaml
codec:
  device: cuda
  num_quantizers: 8

tokenizer:
  model_path: DATASET/tokenize/models/spm.model
```

## Smoke test

```bash
python scripts/raw_to_compiled/convert_phoaudiobook.py \
  --accept-terms \
  --config configs/base.yaml \
  --output DATASET/compiled/phoaudiobook \
  --work-dir DATASET/work/phoaudiobook \
  --split train \
  --max-samples 32 \
  --strict
```

This converts the first 32 rows into the real output. It is not throwaway work; the full command
below continues from row 32.

## Full conversion

After a successful smoke test, resume without `--max-samples`:

```bash
python scripts/raw_to_compiled/convert_phoaudiobook.py \
  --accept-terms \
  --config configs/base.yaml \
  --output DATASET/compiled/phoaudiobook \
  --work-dir DATASET/work/phoaudiobook \
  --split train \
  --resume
```

The source revision is resolved to a commit SHA and stored in `state.json`. If interrupted,
repeat the full command with `--resume`. Completed compiled shards and already encoded rows in
the pending manifest are not processed again.

Useful monitoring commands:

```bash
cat DATASET/work/phoaudiobook/state.json
du -sh DATASET/compiled/phoaudiobook
find DATASET/compiled/phoaudiobook/shards -mindepth 1 -maxdepth 1 -type d | wc -l
```

Add the result to training:

```yaml
train:
  datasets:
    - DATASET/compiled/phoaudiobook
```

Then start training normally:

```bash
python -m donglao_tts.cli.train --config configs/base.yaml --resume none
```

## Compile Emilia English

The Emilia repository is gated. Accept its terms at
<https://huggingface.co/datasets/amphion/Emilia-Dataset> and authenticate first:

```bash
hf auth login
```

The English importer works like the PhoAudiobook importer: it streams source audio, runs English
G2P and the configured MOSS codec, then commits resumable compiled shards. It selects only
`Emilia/EN/*.tar`; Emilia-YODAS is excluded.

`run_en.sh` intentionally uses the same editable layout as `run.sh`:

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/raw_to_compiled/convert_emilia_en.py \
  --accept-terms \
  --config configs/base.yaml \
  --output DATASET/compiled/emilia_en \
  --work-dir DATASET/work/emilia_en \
  --resume
```

Run it from the repository root:

```bash
scripts/raw_to_compiled/run_en.sh
```

`--resume` is safe on the first run when neither output nor state exists. Later runs continue
from `DATASET/work/emilia_en/state.json`. For a smoke test, add `--max-samples 32` immediately
before `--resume` in `run_en.sh`, then remove it for the full conversion.

The source revision is pinned internally, and progress is stored by source tar and sample index.
Raw tar files are streamed instead of retaining the full roughly 1.05 TiB English source
locally.

Before processing English, make sure the SentencePiece model configured at
`tokenizer.model_path` was trained with English phonemes; a Vietnamese-only tokenizer will map
English phonemes to unknown tokens.

## Common failures

- `401`, `403`, or gated-repository errors: accept the terms in the browser, then rerun
  `hf auth login`.
- `TorchCodec` errors: the importer intentionally requests undecoded audio and uses
  `soundfile`; ensure `libsndfile` and the packages in `requirements.txt` are installed.
- CUDA out of memory while encoding: reduce `--g2p-batch-size` only affects G2P RAM, while codec
  encoding is already one audio example at a time. Close other GPU jobs or change
  `codec.device` to `cpu`.
- Interrupted process: never delete `--work-dir`; rerun with exactly the same arguments plus
  `--resume`.
