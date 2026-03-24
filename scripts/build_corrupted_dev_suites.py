"""Build frozen corrupted dev suites for evaluation from a clean dev manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.eval import build_corrupted_dev_suites, load_corrupted_dev_suites_plan


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
        default=Path("configs/corruption/corrupted-dev-suites.toml"),
        help="Corrupted dev suites TOML plan.",
    )
    parser.add_argument(
        "--noise-manifest",
        type=Path,
        default=None,
        help="Optional override for the noise-bank manifest path.",
    )
    parser.add_argument(
        "--rir-manifest",
        type=Path,
        default=None,
        help="Optional override for the RIR-bank manifest path.",
    )
    parser.add_argument(
        "--room-config-manifest",
        type=Path,
        default=None,
        help="Optional override for the room-config manifest path.",
    )
    parser.add_argument(
        "--codec-plan",
        type=Path,
        default=Path("configs/corruption/codec-bank.toml"),
        help="Codec-bank plan used to re-render codec/channel suites.",
    )
    parser.add_argument(
        "--far-field-plan",
        type=Path,
        default=Path("configs/corruption/far-field-bank.toml"),
        help="Far-field plan used to re-render distance suites.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="FFmpeg binary used for codec/channel transforms.",
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
    plan = load_corrupted_dev_suites_plan(args.plan)
    report = build_corrupted_dev_suites(
        project_root=config.paths.project_root,
        plan=plan,
        normalization_config=config.normalization,
        silence_config=config.silence_augmentation,
        plan_path=args.plan,
        noise_manifest_path=args.noise_manifest,
        rir_manifest_path=args.rir_manifest,
        room_config_manifest_path=args.room_config_manifest,
        codec_plan_path=args.codec_plan,
        far_field_plan_path=args.far_field_plan,
        ffmpeg_path=args.ffmpeg_path,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
