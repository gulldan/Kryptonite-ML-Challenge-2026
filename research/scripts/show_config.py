"""Inspect the project configuration with optional overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="Path to the base TOML config.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file with secrets.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="Print resolved secrets without masking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )
    print(json.dumps(config.to_dict(mask_secrets=not args.show_secrets), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
