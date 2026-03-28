"""Build a reproducible PyTorch vs ORT vs TensorRT benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eval import (
    build_backend_benchmark_report,
    load_backend_benchmark_config,
    write_backend_benchmark_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/release/backend-benchmark.toml"),
        help="Path to the backend benchmark TOML config.",
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
    config = load_backend_benchmark_config(config_path=args.config)
    report = build_backend_benchmark_report(config, config_path=args.config)
    written = write_backend_benchmark_report(report)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
    else:
        status = "PASS" if written.summary.passed else "FAIL"
        print(
            "\n".join(
                [
                    f"Backend benchmark report: {status}",
                    f"Model version: {report.model_version or 'unknown'}",
                    (
                        "Backends passed: "
                        f"{written.summary.successful_backend_count}/"
                        f"{written.summary.backend_count}"
                    ),
                    (
                        "Workloads passed: "
                        f"{written.summary.successful_workload_count}/"
                        f"{written.summary.workload_count}"
                    ),
                    f"JSON: {written.report_json_path}",
                    f"Markdown: {written.report_markdown_path}",
                    f"Rows: {written.workload_rows_path}",
                ]
            )
        )

    if not written.summary.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
