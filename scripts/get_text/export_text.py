#!/usr/bin/env python3
"""Export language and original text from donglao-tts compiled datasets."""

import argparse
import csv
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


COMPILED_FORMAT = "donglao-tts-compiled"


def _catalog_paths(input_path):
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        if input_path.name != "catalog.json":
            raise ValueError(f"expected catalog.json, got: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input does not exist: {input_path}")
    direct_catalog = input_path / "catalog.json"
    if direct_catalog.is_file():
        return [direct_catalog]
    catalogs = sorted(input_path.rglob("catalog.json"))
    if not catalogs:
        raise FileNotFoundError(f"no catalog.json found under: {input_path}")
    return catalogs


def _load_catalog(path):
    try:
        with path.open("r", encoding="utf-8") as source:
            catalog = json.load(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if catalog.get("format") != COMPILED_FORMAT:
        raise ValueError(
            f"unsupported compiled format in {path}: {catalog.get('format')!r}"
        )
    language = str(catalog.get("language", "")).strip()
    if not language:
        raise ValueError(f"missing language in {path}")
    if not isinstance(catalog.get("shards"), list):
        raise ValueError(f"missing shards list in {path}")
    return catalog, language


def _metadata_path(dataset_root, shard, catalog_path):
    try:
        relative_path = Path(shard["metadata"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"shard without metadata path in {catalog_path}") from exc
    if relative_path.is_absolute():
        raise ValueError(f"absolute metadata path is not allowed: {relative_path}")
    resolved = (dataset_root / relative_path).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"metadata path escapes dataset root in {catalog_path}: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing metadata file: {resolved}")
    return resolved


def _single_line_text(value):
    if not isinstance(value, str):
        raise TypeError(f"text must be a string, got {type(value).__name__}")
    return " ".join(value.splitlines()).strip()


def iter_language_text(input_path, *, languages=None, on_error="fail", error_stream=None):
    """Yield ``(language, text)`` from one compiled dataset or a directory tree."""
    if on_error not in ("fail", "skip"):
        raise ValueError("on_error must be 'fail' or 'skip'")
    languages = set(languages or ())
    error_stream = error_stream or sys.stderr

    for catalog_path in _catalog_paths(input_path):
        try:
            catalog, catalog_language = _load_catalog(catalog_path)
        except (OSError, ValueError) as exc:
            if on_error == "fail":
                raise
            print(f"skip catalog {catalog_path}: {exc}", file=error_stream)
            continue
        if languages and catalog_language not in languages:
            continue

        dataset_root = catalog_path.parent.resolve()
        for shard in catalog["shards"]:
            try:
                metadata_path = _metadata_path(dataset_root, shard, catalog_path)
            except (OSError, TypeError, ValueError) as exc:
                if on_error == "fail":
                    raise
                print(f"skip shard in {catalog_path}: {exc}", file=error_stream)
                continue

            with metadata_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    try:
                        entry = json.loads(line)
                        language = str(entry.get("language", catalog_language)).strip()
                        if not language:
                            raise ValueError("empty language")
                        if language != catalog_language:
                            raise ValueError(
                                f"record language {language!r} differs from "
                                f"catalog language {catalog_language!r}"
                            )
                        text = _single_line_text(entry["text"])
                        if not text:
                            raise ValueError("empty text")
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        if on_error == "fail":
                            raise ValueError(
                                f"invalid metadata at {metadata_path}:{line_number}: {exc}"
                            ) from exc
                        print(
                            f"skip {metadata_path}:{line_number}: {exc}",
                            file=error_stream,
                        )
                        continue
                    yield language, text


@contextmanager
def _atomic_csv_destination(output_path, *, force=False):
    if str(output_path) == "-":
        yield sys.stdout
        return

    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite {output_path}; pass --force to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as destination:
            yield destination
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def export_text(
    input_path,
    output_path,
    *,
    languages=None,
    deduplicate=False,
    on_error="fail",
    force=False,
    progress_every=100_000,
    error_stream=None,
):
    """Export compiled metadata and return output statistics."""
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    error_stream = error_stream or sys.stderr
    seen = set() if deduplicate else None
    read_count = 0
    written_count = 0

    with _atomic_csv_destination(output_path, force=force) as destination:
        writer = csv.writer(
            destination,
            delimiter="|",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerow(("language", "text"))
        for language, text in iter_language_text(
            input_path,
            languages=languages,
            on_error=on_error,
            error_stream=error_stream,
        ):
            read_count += 1
            key = (language, text)
            if seen is not None:
                if key in seen:
                    continue
                seen.add(key)
            writer.writerow(key)
            written_count += 1
            if progress_every and read_count % progress_every == 0:
                print(
                    f"progress: {read_count:,} read, {written_count:,} written",
                    file=error_stream,
                )

    return {"read": read_count, "written": written_count}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="DATASET/compiled",
        help="compiled dataset, catalog.json, or directory containing compiled datasets",
    )
    parser.add_argument(
        "--output",
        default="DATASET/text/language_text.csv",
        help="destination CSV path, or - for stdout",
    )
    parser.add_argument(
        "--language",
        action="append",
        help="include only this language; repeat for multiple languages",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="write each exact language/text pair once (uses additional memory)",
    )
    parser.add_argument(
        "--on-error",
        choices=("fail", "skip"),
        default="fail",
        help="stop on malformed metadata or report and skip it",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="print progress every N input rows; use 0 to disable",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        stats = export_text(
            args.input,
            args.output,
            languages=args.language,
            deduplicate=args.deduplicate,
            on_error=args.on_error,
            force=args.force,
            progress_every=args.progress_every,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"complete: {stats['read']:,} read, {stats['written']:,} written to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
