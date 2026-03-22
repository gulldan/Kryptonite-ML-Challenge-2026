"""Create a local tracking run and log a small reproducibility artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.repro import build_reproducibility_snapshot
from kryptonite.tracking import build_tracker


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
        "--kind",
        choices=("train", "eval", "export"),
        default="train",
        help="Logical run category to log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )
    tracker = build_tracker(config=config)
    run = tracker.start_run(kind=args.kind, config=config.to_dict(mask_secrets=True))
    run.log_metrics({"loss": 0.42, "eer": 0.11}, step=1)

    snapshot = build_reproducibility_snapshot(config=config, config_path=args.config)
    snapshot_path = run.run_dir / "reproducibility-snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    run.log_artifact(snapshot_path)
    summary = run.finish(summary={"smoke_kind": args.kind})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
