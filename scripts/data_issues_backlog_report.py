"""Generate a reproducible cleanup backlog from the current EDA inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.eda.data_issues_backlog import (
    build_data_issues_backlog_report,
    write_data_issues_backlog_report,
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
        default=Path("artifacts/eda/data-issues-backlog"),
        help="Directory where the JSON and Markdown backlog should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    report = build_data_issues_backlog_report(
        project_root=config.paths.project_root,
        manifests_root=config.paths.manifests_root,
    )
    written = write_data_issues_backlog_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "issue_count": report.issue_count,
                "issue_counts_by_action": report.issue_counts_by_action,
                "issue_counts_by_severity": report.issue_counts_by_severity,
                "stop_rules": report.stop_rules,
                "warnings": report.warnings,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
