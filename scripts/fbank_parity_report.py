"""Compare offline and streaming 80-dim Fbank extraction on a manifest-backed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.features import build_fbank_parity_report, write_fbank_parity_report


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
        default=Path("artifacts/eda/fbank-parity"),
        help="Directory where JSON/Markdown/rows artifacts should be written.",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=float,
        default=137.0,
        help="Streaming chunk size in milliseconds; defaults to a non-hop-aligned value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on analyzed manifest rows for quick smoke runs.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for offline/online parity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    report = build_fbank_parity_report(
        project_root=config.paths.project_root,
        manifest_path=args.manifest,
        normalization=config.normalization,
        vad=config.vad,
        features=config.features,
        chunk_duration_ms=args.chunk_duration_ms,
        limit=args.limit,
        atol=args.atol,
    )
    written = write_fbank_parity_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "manifest_path": report.manifest_path,
                "chunk_duration_ms": report.chunk_duration_ms,
                "summary": report.summary.to_dict(),
                "request": report.to_dict(include_records=False)["request"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
