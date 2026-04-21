"""Validate one or more TOML configs against the repository JSON Schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from kryptonite.config_schema import format_validation_error, validate_config_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="+",
        help="TOML config files to validate.",
    )
    parser.add_argument(
        "--schema",
        default="configs/schema.json",
        help="Path to the JSON Schema file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    schema_path = Path(args.schema)
    has_errors = False
    for config_path_raw in args.configs:
        config_path = Path(config_path_raw)
        errors = validate_config_file(config_path=config_path, schema_path=schema_path)
        if not errors:
            print(f"[schema] ok {config_path}")
            continue
        has_errors = True
        print(f"[schema] fail {config_path}")
        for error in errors:
            print(f"  - {format_validation_error(error)}")
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
