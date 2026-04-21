#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import load_config, runs_root, submissions_root, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run soft CAM++ no-aug and aug experiments with automatic submissions."
    )
    parser.add_argument(
        "--config-noaug", required=True, help="Path to soft no-augmentation config."
    )
    parser.add_argument("--config-aug", required=True, help="Path to soft augmentation config.")
    parser.add_argument("--csv", required=True, help="Path to test_public.csv.")
    parser.add_argument(
        "--mode", default="segment_mean", help="Submission embedding extraction mode."
    )
    parser.add_argument("--topk", type=int, default=10, help="Number of neighbours in submission.")
    parser.add_argument(
        "--run-prefix", default="campp_en_soft_cycle", help="Run prefix for generated names."
    )
    return parser.parse_args()


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def maybe_build_submission(
    python_bin: Path,
    root: Path,
    config_path: Path,
    checkpoint_path: Path,
    csv_path: Path,
    mode: str,
    topk: int,
    run_name: str,
    launcher_log_root: Path,
) -> Path | None:
    if not checkpoint_path.exists():
        return None
    log_path = launcher_log_root / f"{run_name}.log"
    cmd = [
        str(python_bin),
        str(root / "code" / "campp" / "build_submission.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--mode",
        mode,
        "--csv",
        str(csv_path),
        "--topk",
        str(topk),
        "--run-name",
        run_name,
    ]
    run_logged(cmd, log_path)
    return submissions_root(load_config(config_path)) / run_name / "submission.csv"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    python_bin = root / ".venv" / "bin" / "python"
    csv_path = Path(args.csv).resolve()
    launcher_log_root = root / "data" / "campp_runs" / "campp_en_ft" / "launcher_logs"
    launcher_log_root.mkdir(parents=True, exist_ok=True)

    cycle_results: dict[str, dict[str, str | None]] = {}

    phases = [
        ("soft_noaug", Path(args.config_noaug).resolve(), f"{args.run_prefix}_noaug_run001"),
        ("soft_aug", Path(args.config_aug).resolve(), f"{args.run_prefix}_aug_run001"),
    ]

    for phase_name, config_path, run_name in phases:
        config = load_config(config_path)
        train_log = launcher_log_root / f"{run_name}.log"
        train_cmd = [
            str(python_bin),
            str(root / "code" / "campp" / "finetune_campp.py"),
            "--config",
            str(config_path),
            "--run-name",
            run_name,
        ]
        run_logged(train_cmd, train_log)

        run_root = runs_root(config) / run_name
        checkpoint_root = run_root / "checkpoints"
        epoch1_name = f"submission_{run_name}_epoch001_test_public"
        best_name = f"submission_{run_name}_best_p10_test_public"
        epoch1_submission = maybe_build_submission(
            python_bin=python_bin,
            root=root,
            config_path=config_path,
            checkpoint_path=checkpoint_root / "epoch_001.pt",
            csv_path=csv_path,
            mode=args.mode,
            topk=args.topk,
            run_name=epoch1_name,
            launcher_log_root=launcher_log_root,
        )
        best_submission = maybe_build_submission(
            python_bin=python_bin,
            root=root,
            config_path=config_path,
            checkpoint_path=checkpoint_root / "best_p10.pt",
            csv_path=csv_path,
            mode=args.mode,
            topk=args.topk,
            run_name=best_name,
            launcher_log_root=launcher_log_root,
        )
        cycle_results[phase_name] = {
            "run_name": run_name,
            "run_root": str(run_root),
            "epoch1_submission": str(epoch1_submission) if epoch1_submission else None,
            "best_submission": str(best_submission) if best_submission else None,
        }

    summary_path = root / "data" / "campp_runs" / "campp_en_ft" / "soft_cycle_summary.json"
    write_json(summary_path, cycle_results)
    print(json.dumps(cycle_results, ensure_ascii=False, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
