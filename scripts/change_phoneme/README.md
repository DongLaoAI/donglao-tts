# Migrate compiled phonemes to donglao-g2p

This migration preserves the source under `DATASET/compiled` and produces training-ready
datasets under the intentionally named `DATASET/complied_v1`.

The migration has three required stages because compiled `text_ids.npy` and `rows.npy` depend on
the SentencePiece model:

```bash
scripts/change_phoneme/01_prepare.sh
scripts/change_phoneme/02_build_sentencepiece.sh
scripts/change_phoneme/03_finalize.sh
```

Or run all stages:

```bash
scripts/change_phoneme/run.sh
```

Stop any raw-to-compiled importer before starting stage 1 and keep it stopped until stage 3
finishes. The work plan freezes catalog identities so a growing or modified source cannot be
silently mixed into the migration. If the source changes after a plan is created, choose a new
work directory and output root for a fresh snapshot.

Outputs:

- `DATASET/change_phoneme_v1/`: resumable prepared metadata and per-shard error reports.
- `DATASET/tokenize/phonemes_v1.txt`: one donglao-g2p phoneme sentence per line.
- `DATASET/tokenize/models_v1/spm.model`: the new SentencePiece model.
- `DATASET/tokenize/models_v1/spm.vocab`: the new vocabulary.
- `DATASET/complied_v1/<corpus>/`: rebuilt compiled datasets using the new model.

Codec arrays are hardlinked where the filesystem supports hardlinks, so this does not duplicate
the multi-gigabyte codec payload. All metadata, text IDs, row indexes, tokenizer hashes, and G2P
profile metadata are rebuilt. Invalid text, failed G2P records, empty phonemes, and outputs
containing `<unk>` are logged and skipped by the default preparation command.

After migration, update the training configuration:

```yaml
tokenizer:
  model_path: DATASET/tokenize/models_v1/spm.model

train:
  datasets:
    - DATASET/complied_v1/emilia_en
    - DATASET/complied_v1/libritts100
    - DATASET/complied_v1/phoaudiobook
    - DATASET/complied_v1/vieneu
```

Do not train from `DATASET/change_phoneme_v1`; it is intermediate metadata, not a compiled
dataset.
