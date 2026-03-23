"""Generate a reproducible audio-quality EDA report from the current manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.eda.dataset_audio_quality import (
    build_dataset_audio_quality_report,
    write_dataset_audio_quality_report,
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
        default=Path("artifacts/eda/dataset-audio-quality"),
        help="Directory where the JSON and Markdown report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    report = build_dataset_audio_quality_report(
        project_root=config.paths.project_root,
        manifests_root=config.paths.manifests_root,
    )
    written = write_dataset_audio_quality_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "manifest_count": report.manifest_count,
                "raw_entry_count": report.raw_entry_count,
                "deduplicated_entry_count": report.total_summary.entry_count,
                "waveform_metrics_count": report.total_summary.waveform_metrics_count,
                "flag_counts": report.total_summary.flag_counts,
                "warnings": report.warnings,
                "patterns": [pattern.to_dict() for pattern in report.patterns],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
