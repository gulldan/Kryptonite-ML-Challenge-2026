"""Compare optional VAD/trimming modes on a manifest-backed dev split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.eda.vad_trimming import build_vad_trimming_report, write_vad_trimming_report


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
        help="Manifest JSONL to profile, usually the held-out dev manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eda/vad-trimming"),
        help="Directory where the JSON/Markdown/rows artifacts should be written.",
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
    report = build_vad_trimming_report(
        project_root=config.paths.project_root,
        manifest_path=args.manifest,
        normalization=config.normalization,
        vad=config.vad,
        limit=args.limit,
    )
    written = write_vad_trimming_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "manifest_path": report.manifest_path,
                "row_count": report.row_count,
                "modes": list(report.modes),
                "backend": report.backend,
                "provider": report.provider,
                "min_output_duration_seconds": report.min_output_duration_seconds,
                "min_retained_ratio": report.min_retained_ratio,
                "summaries": [summary.to_dict() for summary in report.summaries],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
