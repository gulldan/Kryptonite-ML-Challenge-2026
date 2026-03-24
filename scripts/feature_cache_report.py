"""Build a reproducible feature-cache benchmark and policy report from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.features import (
    build_feature_cache_benchmark_report,
    write_feature_cache_benchmark_report,
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
        default=Path("artifacts/eda/feature-cache"),
        help="Directory where JSON/Markdown/rows artifacts should be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on analyzed manifest rows for quick smoke runs.",
    )
    parser.add_argument(
        "--benchmark-device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Optional override for feature cache benchmark device selection.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite cache entries even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    report = build_feature_cache_benchmark_report(
        project_root=config.paths.project_root,
        cache_root=config.paths.cache_root,
        manifest_path=args.manifest,
        normalization=config.normalization,
        vad=config.vad,
        features=config.features,
        feature_cache=config.feature_cache,
        limit=args.limit,
        benchmark_device=args.benchmark_device,
        force=args.force,
    )
    written = write_feature_cache_benchmark_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "manifest_path": report.manifest_path,
                "selected_device": report.selected_device,
                "summary": report.materialization.summary.to_dict(),
                "benchmarks": [scenario.to_dict() for scenario in report.benchmarks],
                "policy": report.policy.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
