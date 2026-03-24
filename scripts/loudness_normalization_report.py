"""Compare loader-time loudness normalization against the baseline waveform path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.eda.loudness_normalization import (
    build_loudness_normalization_report,
    write_loudness_normalization_report,
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
        default=Path("artifacts/eda/loudness-normalization"),
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
    report = build_loudness_normalization_report(
        project_root=config.paths.project_root,
        manifest_path=args.manifest,
        normalization=config.normalization,
        vad=config.vad,
        limit=args.limit,
    )
    written = write_loudness_normalization_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "manifest_path": report.manifest_path,
                "loudness_mode": report.loudness_mode,
                "target_loudness_dbfs": report.target_loudness_dbfs,
                "max_loudness_gain_db": report.max_loudness_gain_db,
                "max_loudness_attenuation_db": report.max_loudness_attenuation_db,
                "vad_mode": report.vad_mode,
                "limit": report.limit,
                "summary": report.summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
