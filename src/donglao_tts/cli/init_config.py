"""Write the packaged example configuration to a user-selected path."""

import argparse
import os
from importlib.resources import files
from pathlib import Path

from donglao_tts.cli._io import atomic_text_writer


def _load_template():
    packaged_template = files("donglao_tts.resources").joinpath("base.yaml")
    try:
        return packaged_template.read_text(encoding="utf-8")
    except FileNotFoundError:
        # Hatch places the canonical template in the wheel. During development,
        # use that same source file before a wheel has been built.
        source_template = Path(__file__).resolve().parents[3] / "configs" / "base.yaml"
        return source_template.read_text(encoding="utf-8")


def initialize_config(output_path, force=False):
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not force:
        raise FileExistsError(f"refusing to overwrite existing config: {output_path}")

    template = _load_template()
    with atomic_text_writer(output_path) as destination:
        destination.write(template)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="destination YAML path, for example configs/local.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    output = initialize_config(args.output, force=args.force)
    print(f"wrote example configuration to {output}")


if __name__ == "__main__":
    main()
