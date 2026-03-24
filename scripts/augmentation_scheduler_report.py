"""Build a manifest-backed coverage report for the augmentation scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.training import (
    build_augmentation_scheduler_report,
    write_augmentation_scheduler_report,
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
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/augmentation-scheduler"),
        help="Directory where JSON/Markdown/epoch artifacts should be written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of epochs to simulate.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=512,
        help="Number of recipes to sample per epoch when estimating coverage.",
    )
    parser.add_argument(
        "--noise-manifest",
        type=Path,
        default=None,
        help="Optional override for the noise-bank manifest JSONL.",
    )
    parser.add_argument(
        "--room-config-manifest",
        type=Path,
        default=None,
        help="Optional override for the RIR room-config JSONL.",
    )
    parser.add_argument(
        "--distance-manifest",
        type=Path,
        default=None,
        help="Optional override for the far-field manifest JSONL.",
    )
    parser.add_argument(
        "--codec-manifest",
        type=Path,
        default=None,
        help="Optional override for the codec-bank manifest JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    report = build_augmentation_scheduler_report(
        project_root=config.paths.project_root,
        scheduler_config=config.augmentation_scheduler,
        silence_config=config.silence_augmentation,
        total_epochs=args.epochs or config.training.max_epochs,
        samples_per_epoch=args.samples_per_epoch,
        seed=config.runtime.seed,
        noise_manifest_path=args.noise_manifest,
        room_config_manifest_path=args.room_config_manifest,
        distance_manifest_path=args.distance_manifest,
        codec_manifest_path=args.codec_manifest,
    )
    written = write_augmentation_scheduler_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "summary": report.summary.to_dict(),
                "manifest_paths": report.manifest_paths.to_dict(),
                "catalog": report.catalog.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
