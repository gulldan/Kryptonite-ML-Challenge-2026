"""Run the release-oriented inference stress matrix and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.serve import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_BENCHMARK_ITERATIONS,
    DEFAULT_VERIFY_THRESHOLD,
    DEFAULT_WARMUP_ITERATIONS,
    build_inference_stress_report,
    default_stress_report_output_root,
    write_inference_stress_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment/infer.toml"),
        help="Path to the active serving config.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file with secrets.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to artifacts/inference-stress/report.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_sizes",
        action="append",
        type=int,
        default=None,
        help="Burst size to benchmark. Repeat to validate multiple batch sizes.",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=DEFAULT_BENCHMARK_ITERATIONS,
        help="How many benchmark iterations to execute per burst size.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
        help="How many warmup iterations to execute before timing.",
    )
    parser.add_argument(
        "--verify-threshold",
        type=float,
        default=DEFAULT_VERIFY_THRESHOLD,
        help="Decision threshold used for the verify scenarios.",
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
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )
    report = build_inference_stress_report(
        config=config,
        batch_sizes=tuple(args.batch_sizes or DEFAULT_BATCH_SIZES),
        benchmark_iterations=args.benchmark_iterations,
        warmup_iterations=args.warmup_iterations,
        verify_threshold=args.verify_threshold,
    )
    written = write_inference_stress_report(
        report,
        output_root=(
            default_stress_report_output_root(config=config)
            if args.output_root is None
            else args.output_root
        ),
    )

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
    else:
        status = "PASS" if report.summary.passed else "FAIL"
        print(
            "\n".join(
                [
                    f"Inference stress report: {status}",
                    f"Backend: {report.service.selected_backend} / {report.service.implementation}",
                    f"Model version: {report.service.model_version}",
                    f"Largest validated burst: {report.hard_limits.largest_validated_batch_size}",
                    f"Peak process RSS MiB: {report.memory.peak_process_rss_mib}",
                    f"Peak CUDA allocated MiB: {report.memory.peak_cuda_allocated_mib}",
                    f"JSON: {written.report_json_path}",
                    f"Markdown: {written.report_markdown_path}",
                ]
            )
        )

    if not report.summary.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
