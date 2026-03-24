"""Build a manifest-driven ablation report for silence and pause augmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.eda.silence_augmentation import (
    build_silence_augmentation_report,
    write_silence_augmentation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="Path to the base TOML config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest JSONL to analyze, usually the held-out dev manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eda/silence-augmentation"),
        help="Directory where JSON/Markdown/rows artifacts should be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on analyzed manifest rows for quick smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    report = build_silence_augmentation_report(
        project_root=config.paths.project_root,
        manifest_path=args.manifest,
        normalization=config.normalization,
        vad=config.vad,
        silence_augmentation=config.silence_augmentation,
        seed=config.runtime.seed,
        limit=args.limit,
    )
    written = write_silence_augmentation_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "manifest_path": report.manifest_path,
                "seed": report.seed,
                "limit": report.limit,
                "vad_mode": report.vad_mode,
                "config": report.to_dict()["config"],
                "summary": report.summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
