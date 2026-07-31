# Export compiled text

Export the original text and language from one or more donglao-tts compiled datasets without
reading codec arrays.

Run from the repository root:

```bash
scripts/get_text/run.sh
```

The default output is:

```text
DATASET/text/language_text.csv
```

Its pipe-delimited CSV format is:

```csv
language|text
en|Hello world.
vi|Xin chào.
```

Fields containing `|` are quoted according to CSV rules. Newlines inside source text are
normalized to spaces so every record occupies one physical line.

Useful commands:

```bash
# English only
python3 scripts/get_text/export_text.py \
  --input DATASET/compiled \
  --output DATASET/text/en.csv \
  --language en

# Vietnamese and English, removing exact duplicates
python3 scripts/get_text/export_text.py \
  --input DATASET/compiled \
  --output DATASET/text/unique.csv \
  --language vi \
  --language en \
  --deduplicate

# One compiled corpus
python3 scripts/get_text/export_text.py \
  --input DATASET/compiled/emilia_en \
  --output DATASET/text/emilia_en.csv
```
