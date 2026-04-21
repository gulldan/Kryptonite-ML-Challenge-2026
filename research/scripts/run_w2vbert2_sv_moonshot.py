"""Run the staged w2v-BERT 2.0 SV moonshot: LoRA+Adapter -> full FT -> LMFT."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    started_at = time.time()
    print(
        f"[{args.run_id}] start {datetime.now(UTC).isoformat()} gpu={args.gpu_label}",
        flush=True,
    )
    if args.stage2_checkpoint:
        if args.stage1_checkpoint:
            stage1 = _reuse_stage1(args)
            print(
                f"[moonshot] complete stage1-lora-adapter checkpoint={stage1['checkpoint_path']} "
                f"output_root={stage1['output_root']}",
                flush=True,
            )
        else:
            stage1 = _build_skipped_stage(config_path=args.stage1_config)
            print("[moonshot] skip stage1-lora-adapter via stage2 checkpoint resume", flush=True)
        stage2 = _reuse_stage2(args)
    else:
        if args.stage1_checkpoint:
            stage1 = _reuse_stage1(args)
        else:
            stage1 = _run_stage1(args)
        print(
            f"[moonshot] complete stage1-lora-adapter checkpoint={stage1['checkpoint_path']} "
            f"output_root={stage1['output_root']}",
            flush=True,
        )
        stage2 = _run_stage2(args, stage1_checkpoint=stage1["checkpoint_path"])
    print(
        f"[moonshot] complete stage2-joint-ft checkpoint={stage2['checkpoint_path']} "
        f"output_root={stage2['output_root']}",
        flush=True,
    )
    stage3 = _run_stage3(args, stage2_checkpoint=stage2["checkpoint_path"])
    print(
        f"[moonshot] complete stage3-lmft checkpoint={stage3['checkpoint_path']} "
        f"output_root={stage3['output_root']}",
        flush=True,
    )
    summary = {
        "run_id": args.run_id,
        "started_at_utc": datetime.fromtimestamp(started_at, UTC).isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-label", default="")
    parser.add_argument(
        "--stage1-config",
        default="research/configs/training/w2vbert2-mfa-lora-stage1.toml",
    )
    parser.add_argument(
        "--stage2-config",
        default="research/configs/training/w2vbert2-mfa-joint-ft-stage2.toml",
    )
    parser.add_argument(
        "--stage3-config",
        default="research/configs/training/w2vbert2-mfa-lmft-stage3.toml",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/reports/w2vbert2/W2V1_w2vbert2_mfa_lora_lmft_summary.json",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        default="",
        help="Existing stage1 checkpoint dir/run dir/checkpoint_metadata.json to reuse.",
    )
    parser.add_argument(
        "--stage2-checkpoint",
        default="",
        help="Existing stage2 checkpoint dir/run dir/checkpoint_metadata.json to reuse.",
    )
    return parser.parse_args()


def _run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    payload = _run_json(
        [
            sys.executable,
            "research/scripts/run_teacher_peft.py",
            "--config",
            args.stage1_config,
            "--output",
            "json",
        ],
        label="stage1-lora-adapter",
    )
    return {
        "config": args.stage1_config,
        "checkpoint_path": payload["checkpoint_path"],
        "output_root": payload["output_root"],
        "training_summary_path": payload["training_summary_path"],
        "score_summary_path": payload["score_summary_path"],
        "report_path": payload["report_path"],
    }


def _reuse_stage1(args: argparse.Namespace) -> dict[str, Any]:
    return _reuse_stage(
        config_path=args.stage1_config,
        checkpoint_path=args.stage1_checkpoint,
        label="stage1",
    )


def _reuse_stage2(args: argparse.Namespace) -> dict[str, Any]:
    payload = _reuse_stage(
        config_path=args.stage2_config,
        checkpoint_path=args.stage2_checkpoint,
        label="stage2",
    )
    payload["init_checkpoint"] = args.stage1_checkpoint
    return payload


def _run_stage2(args: argparse.Namespace, *, stage1_checkpoint: str) -> dict[str, Any]:
    payload = _run_json(
        [
            sys.executable,
            "research/scripts/run_teacher_peft_finetune.py",
            "--config",
            args.stage2_config,
            "--init-checkpoint",
            stage1_checkpoint,
            "--merge-lora",
            "--init-classifier-from-checkpoint",
            "--output",
            "json",
        ],
        label="stage2-joint-ft",
    )
    return {
        "config": args.stage2_config,
        "init_checkpoint": stage1_checkpoint,
        "checkpoint_path": payload["checkpoint_path"],
        "output_root": payload["output_root"],
        "training_summary_path": payload["training_summary_path"],
        "score_summary_path": payload["score_summary_path"],
        "report_path": payload["report_path"],
    }


def _run_stage3(args: argparse.Namespace, *, stage2_checkpoint: str) -> dict[str, Any]:
    payload = _run_json(
        [
            sys.executable,
            "research/scripts/run_teacher_peft_finetune.py",
            "--config",
            args.stage3_config,
            "--init-checkpoint",
            stage2_checkpoint,
            "--init-classifier-from-checkpoint",
            "--output",
            "json",
        ],
        label="stage3-lmft",
    )
    return {
        "config": args.stage3_config,
        "init_checkpoint": stage2_checkpoint,
        "checkpoint_path": payload["checkpoint_path"],
        "output_root": payload["output_root"],
        "training_summary_path": payload["training_summary_path"],
        "score_summary_path": payload["score_summary_path"],
        "report_path": payload["report_path"],
    }


def _run_json(command: list[str], *, label: str) -> dict[str, Any]:
    print(f"[moonshot] run {label}: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stderr is None or process.stdout is None:
        raise RuntimeError("Failed to create subprocess pipes for moonshot stage.")
    stderr_chunks: list[str] = []
    for line in process.stderr:
        stderr_chunks.append(line)
        print(line, file=sys.stderr, end="", flush=True)
    stdout = process.stdout.read()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
            output=stdout,
            stderr="".join(stderr_chunks),
        )
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} did not return JSON. stdout was:\n{stdout}") from error


def _resolve_checkpoint_dir(checkpoint_path: str) -> Path:
    candidate = Path(checkpoint_path)
    if candidate.is_file():
        return candidate.parent
    metadata_path = candidate / "checkpoint_metadata.json"
    if metadata_path.is_file():
        return candidate
    nested_checkpoint_dir = candidate / "teacher_peft"
    if (nested_checkpoint_dir / "checkpoint_metadata.json").is_file():
        return nested_checkpoint_dir
    return candidate


def _reuse_stage(
    *,
    config_path: str,
    checkpoint_path: str,
    label: str,
) -> dict[str, Any]:
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)
    output_root = checkpoint_dir.parent
    training_summary_path = output_root / "training_summary.json"
    score_summary_path = output_root / "score_summary.json"
    report_path = output_root / "teacher_peft_report.md"
    print(
        f"[moonshot] reuse {label} checkpoint={checkpoint_dir} output_root={output_root}",
        flush=True,
    )
    return {
        "config": config_path,
        "checkpoint_path": str(checkpoint_dir),
        "output_root": str(output_root),
        "training_summary_path": str(training_summary_path)
        if training_summary_path.is_file()
        else "",
        "score_summary_path": str(score_summary_path) if score_summary_path.is_file() else "",
        "report_path": str(report_path) if report_path.is_file() else "",
        "reused": True,
        "source_checkpoint": str(Path(checkpoint_path)),
    }


def _build_skipped_stage(*, config_path: str) -> dict[str, Any]:
    return {
        "config": config_path,
        "checkpoint_path": "",
        "output_root": "",
        "training_summary_path": "",
        "score_summary_path": "",
        "report_path": "",
        "skipped": True,
    }


if __name__ == "__main__":
    main()
