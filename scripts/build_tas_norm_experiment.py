"""Build a reproducible TAS-norm experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eval import (
    build_tas_norm_experiment_report,
    load_tas_norm_experiment_config,
    write_tas_norm_experiment_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the TAS-norm experiment.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="CLI output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_tas_norm_experiment_config(config_path=args.config)
    built = build_tas_norm_experiment_report(config, config_path=args.config)
    written = write_tas_norm_experiment_report(built)
    report = built.report

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "TAS-norm experiment report complete",
        f"Title: {report.title}",
        f"Report id: {report.report_id}",
        f"Decision: {report.summary.decision}",
        f"Eval winner: {report.summary.eval_winner}",
        f"Train trials: {report.split.train_trial_count}",
        f"Eval trials: {report.split.eval_trial_count}",
        f"Failed checks: {report.summary.failed_check_count}",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
