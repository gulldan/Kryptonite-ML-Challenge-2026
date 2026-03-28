"""Build a reproducible ONNX Runtime parity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eval import (
    build_onnx_parity_report,
    load_onnx_parity_config,
    write_onnx_parity_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the ONNX Runtime parity workflow.",
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
    config = load_onnx_parity_config(config_path=args.config)
    report = build_onnx_parity_report(config, config_path=args.config)
    written = write_onnx_parity_report(report)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "ONNX Runtime parity report complete",
        f"Title: {report.title}",
        f"Report id: {report.report_id}",
        f"Status: {'pass' if written.summary.passed else 'fail'}",
        f"Variants passed: {written.summary.passed_variant_count}/{written.summary.variant_count}",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
        f"Audio rows: {written.audio_rows_path}",
        f"Trial rows: {written.trial_rows_path}",
        f"Metadata promotion applied: {str(written.promotion.applied).lower()}",
    ]
    if written.promotion.error:
        lines.append(f"Metadata promotion error: {written.promotion.error}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
