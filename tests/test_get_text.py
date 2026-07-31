import csv
import json
import sys
from pathlib import Path


GET_TEXT = Path(__file__).resolve().parents[1] / "scripts" / "get_text"
sys.path.insert(0, str(GET_TEXT))
import export_text  # noqa: E402


def _compiled_dataset(root, corpus, language, texts):
    dataset = root / corpus
    shard = dataset / "shards" / "shard-00000"
    shard.mkdir(parents=True)
    metadata = shard / "metadata.jsonl"
    with metadata.open("w", encoding="utf-8") as destination:
        for index, text in enumerate(texts):
            destination.write(
                json.dumps(
                    {
                        "utterance_id": f"{corpus}:{index}",
                        "language": language,
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (dataset / "catalog.json").write_text(
        json.dumps(
            {
                "format": "donglao-tts-compiled",
                "version": 2,
                "corpus": corpus,
                "language": language,
                "shards": [
                    {
                        "name": "shard-00000",
                        "metadata": "shards/shard-00000/metadata.jsonl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_export_text_recurses_and_writes_pipe_csv(tmp_path):
    compiled = tmp_path / "compiled"
    _compiled_dataset(compiled, "english", "en", ["Hello\nworld", "A | B"])
    _compiled_dataset(compiled, "vietnamese", "vi", ["Xin chào"])
    output = tmp_path / "language_text.csv"

    stats = export_text.export_text(compiled, output, progress_every=0)

    with output.open(encoding="utf-8", newline="") as source:
        rows = list(csv.reader(source, delimiter="|"))
    assert rows == [
        ["language", "text"],
        ["en", "Hello world"],
        ["en", "A | B"],
        ["vi", "Xin chào"],
    ]
    assert stats == {"read": 3, "written": 3}


def test_export_text_filters_and_deduplicates(tmp_path):
    compiled = tmp_path / "compiled"
    _compiled_dataset(compiled, "english", "en", ["Same", "Same"])
    _compiled_dataset(compiled, "vietnamese", "vi", ["Giống"])
    output = tmp_path / "english.csv"

    stats = export_text.export_text(
        compiled,
        output,
        languages=["en"],
        deduplicate=True,
        progress_every=0,
    )

    assert output.read_text(encoding="utf-8") == "language|text\nen|Same\n"
    assert stats == {"read": 2, "written": 1}


def test_export_text_skip_bad_record(tmp_path):
    compiled = tmp_path / "compiled"
    _compiled_dataset(compiled, "english", "en", ["Good"])
    metadata = compiled / "english" / "shards" / "shard-00000" / "metadata.jsonl"
    with metadata.open("a", encoding="utf-8") as destination:
        destination.write('{"language":"en","text":""}\n')
        destination.write("not json\n")
    output = tmp_path / "output.csv"

    stats = export_text.export_text(
        compiled,
        output,
        on_error="skip",
        progress_every=0,
    )

    assert output.read_text(encoding="utf-8") == "language|text\nen|Good\n"
    assert stats == {"read": 1, "written": 1}
