"""Build and validate a TensorRT FP16 engine from the promoted ONNX bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.serve.tensorrt_engine import (
    build_tensorrt_fp16_report,
    write_tensorrt_fp16_report,
)
from kryptonite.serve.tensorrt_engine_config import load_tensorrt_fp16_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TensorRT FP16 workflow config.",
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
    config = load_tensorrt_fp16_config(config_path=args.config)
    report = build_tensorrt_fp16_report(config, config_path=args.config)
    written = write_tensorrt_fp16_report(report)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "TensorRT FP16 engine workflow complete",
        f"Status: {'pass' if written.summary.passed else 'fail'}",
        f"Output root: {written.output_root}",
        f"Report JSON: {written.report_json_path}",
        f"Report Markdown: {written.report_markdown_path}",
        f"Metadata promotion applied: {str(written.promotion.applied).lower()}",
    ]
    if written.promotion.error:
        lines.append(f"Metadata promotion error: {written.promotion.error}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
