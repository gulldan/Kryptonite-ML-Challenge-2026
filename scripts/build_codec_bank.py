"""Assemble deterministic codec/channel simulation presets into preview artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.data.codec_bank import (
    build_codec_bank,
    load_codec_bank_plan,
    write_codec_bank_report,
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
        "--plan",
        type=Path,
        default=Path("configs/corruption/codec-bank.toml"),
        help="Codec-bank TOML plan with deterministic FFmpeg preset definitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/corruptions/codec-bank"),
        help="Directory where preview audio, manifests, and reports should be written.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="FFmpeg executable used to render previews.",
    )
    parser.add_argument(
        "--ffprobe-path",
        default="ffprobe",
        help="FFprobe executable used to record environment metadata.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    plan = load_codec_bank_plan(args.plan)
    report = build_codec_bank(
        project_root=config.paths.project_root,
        output_root=args.output_dir,
        plan=plan,
        ffmpeg_path=args.ffmpeg_path,
        ffprobe_path=args.ffprobe_path,
        plan_path=args.plan,
    )
    written = write_codec_bank_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "summary": report.summary.to_dict(),
                "ffmpeg": report.ffmpeg.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
