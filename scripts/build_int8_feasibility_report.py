"""Build a reproducible INT8 feasibility report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eval import (
    build_int8_feasibility_report,
    load_int8_feasibility_config,
    write_int8_feasibility_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the INT8 feasibility decision.",
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
    config = load_int8_feasibility_config(config_path=args.config)
    report = build_int8_feasibility_report(config, config_path=args.config)
    written = write_int8_feasibility_report(report)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "INT8 feasibility report complete",
        f"Title: {report.title}",
        f"Report id: {report.report_id}",
        f"Decision: {report.summary.decision}",
        f"Calibration inputs: {report.calibration_set.selected_input_count}",
        f"Failed checks: {report.summary.failed_check_count}",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
