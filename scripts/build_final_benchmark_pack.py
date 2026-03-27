"""Build a self-contained final benchmark pack from frozen candidate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eval import (
    build_final_benchmark_pack,
    load_final_benchmark_pack_config,
    write_final_benchmark_pack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the candidate pack.",
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
    config = load_final_benchmark_pack_config(config_path=args.config)
    report = build_final_benchmark_pack(config, config_path=args.config)
    written = write_final_benchmark_pack(report)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    print(
        "\n".join(
            [
                "Final benchmark pack complete",
                f"Title: {report.title}",
                f"Output root: {written.output_root}",
                f"Candidates: {written.summary.candidate_count}",
                f"Best EER candidate: {written.summary.best_eer_candidate_id}",
                f"Lowest latency candidate: {written.summary.lowest_latency_candidate_id}",
                f"JSON: {written.report_json_path}",
                f"Markdown: {written.report_markdown_path}",
            ]
        )
    )


if __name__ == "__main__":
    main()
