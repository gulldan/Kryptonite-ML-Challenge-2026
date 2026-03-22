"""Run a reproducibility probe or compare two short seeded runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.repro import build_reproducibility_snapshot, run_reproducibility_self_check


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
        "--self-check",
        action="store_true",
        help="Run the subprocess-based self-check instead of a single probe.",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Emit the current process snapshot as JSON for internal subprocess use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )

    if args.self_check:
        result = run_reproducibility_self_check(
            script_path=Path(__file__),
            config_path=args.config,
            env_file=args.env_file,
            overrides=args.override,
            config=config,
        )
    else:
        result = build_reproducibility_snapshot(config=config, config_path=args.config)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
