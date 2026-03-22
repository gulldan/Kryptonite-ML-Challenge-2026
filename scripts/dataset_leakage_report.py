"""Generate a reproducible duplicate/leakage audit from the current manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.eda.dataset_leakage import (
    build_dataset_leakage_report,
    write_dataset_leakage_report,
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
        default=Path("artifacts/eda/dataset-leakage"),
        help="Directory where the JSON and Markdown report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    report = build_dataset_leakage_report(
        project_root=config.paths.project_root,
        manifests_root=config.paths.manifests_root,
    )
    written = write_dataset_leakage_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "raw_record_count": report.raw_record_count,
                "deduplicated_record_count": report.deduplicated_record_count,
                "trial_count": report.trial_count,
                "finding_count": report.finding_count,
                "finding_counts_by_severity": report.finding_counts_by_severity,
                "warnings": report.warnings,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
