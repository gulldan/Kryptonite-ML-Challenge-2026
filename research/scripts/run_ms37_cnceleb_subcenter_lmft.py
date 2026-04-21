"""Orchestrate MS37: MS31 -> CN-Celeb mixed sub-center adaptation -> LMFT -> C4 tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    started_at = time.time()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"[{args.run_id}] start {datetime.now(UTC).isoformat()} gpu={gpu}",
        flush=True,
    )
    _wait_for_cnceleb(args)
    manifest_summary = _build_cnceleb_manifest(args)

    stage_a_checkpoint = _run_finetune_stage(
        args,
        label="stage-a",
        config=args.stage_a_config,
        init_checkpoint=args.ms31_checkpoint,
        output_root=args.stage_a_output_root,
        init_classifier=False,
    )
    stage_b_checkpoint = _run_finetune_stage(
        args,
        label="stage-b-lmft",
        config=args.stage_b_config,
        init_checkpoint=stage_a_checkpoint,
        output_root=args.stage_b_output_root,
        init_classifier=True,
    )
    tail_summary = _run_public_tail(args, checkpoint=stage_b_checkpoint)
    run_summary = {
        "run_id": args.run_id,
        "started_at_utc": datetime.fromtimestamp(started_at, UTC).isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "ms31_checkpoint": args.ms31_checkpoint,
        "manifest_summary": manifest_summary,
        "stage_a_checkpoint": stage_a_checkpoint,
        "stage_b_checkpoint": stage_b_checkpoint,
        "tail_summary": tail_summary,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cnceleb-root", default="datasets/CN-Celeb_flac")
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--wait-poll-seconds", type=int, default=600)
    parser.add_argument("--manifest-output-dir", default="artifacts/manifests/cnceleb_v2_ms37")
    parser.add_argument("--manifest-experiment-id", default="cnceleb_v2_ms37")
    parser.add_argument(
        "--base-train-manifest",
        default="artifacts/manifests/participants_fixed/train_manifest.jsonl",
    )
    parser.add_argument("--min-cnceleb-speakers", type=int, default=100)
    parser.add_argument("--min-cnceleb-train-rows", type=int, default=10000)
    parser.add_argument(
        "--ms31-checkpoint",
        default=(
            "artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/"
            "20260413T203245Z-b87036ccb3db/campp_encoder.pt"
        ),
    )
    parser.add_argument(
        "--stage-a-config",
        default="research/configs/training/campp-ms37-cnceleb-mixed-subcenter-lowlr.toml",
    )
    parser.add_argument(
        "--stage-b-config",
        default="research/configs/training/campp-ms37-cnceleb-mixed-subcenter-lmft.toml",
    )
    parser.add_argument(
        "--stage-a-output-root",
        default="artifacts/baselines/campp-ms37-cnceleb-mixed-subcenter-lowlr",
    )
    parser.add_argument(
        "--stage-b-output-root",
        default="artifacts/baselines/campp-ms37-cnceleb-mixed-subcenter-lmft",
    )
    parser.add_argument(
        "--public-manifest-csv",
        default="artifacts/eda/participants_public_baseline/test_public_manifest.csv",
    )
    parser.add_argument("--template-csv", default="datasets/Для участников/test_public.csv")
    parser.add_argument("--data-root", default="datasets/Для участников")
    parser.add_argument(
        "--tail-output-dir",
        default="artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms37_cnceleb_subcenter_lmft",
    )
    parser.add_argument(
        "--frontend-pack-dir", default="artifacts/cache/campp-official-public-ms1-v1-pack"
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/reports/ms37/MS37_campp_ms31_cnceleb_mixed_subcenter_lmft_summary.json",
    )
    return parser.parse_args()


def _wait_for_cnceleb(args: argparse.Namespace) -> None:
    root = Path(args.cnceleb_root)
    deadline = time.time() + args.wait_timeout_seconds
    while True:
        if root.is_dir():
            print(f"[{args.run_id}] CN-Celeb root ready: {root}", flush=True)
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {root}")
        archive = Path("datasets/cn-celeb_v2.tar.gz")
        archive_size = archive.stat().st_size if archive.exists() else 0
        print(
            f"[{args.run_id}] waiting for {root}; archive_size_gib={archive_size / 1024**3:.2f}",
            flush=True,
        )
        time.sleep(args.wait_poll_seconds)


def _build_cnceleb_manifest(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "research/scripts/build_cnceleb_manifests.py",
        "--root",
        args.cnceleb_root,
        "--output-dir",
        args.manifest_output_dir,
        "--experiment-id",
        args.manifest_experiment_id,
        "--base-train-manifest",
        args.base_train_manifest,
    ]
    _run(command, label="build-cnceleb-manifests")
    summary_path = Path(args.manifest_output_dir) / f"{args.manifest_experiment_id}_summary.json"
    summary = json.loads(summary_path.read_text())
    if int(summary["speaker_count"]) < args.min_cnceleb_speakers:
        raise ValueError(f"CN-Celeb speaker guard failed: {summary['speaker_count']}")
    if int(summary["train_row_count"]) < args.min_cnceleb_train_rows:
        raise ValueError(f"CN-Celeb train-row guard failed: {summary['train_row_count']}")
    print(f"[{args.run_id}] manifest summary: {json.dumps(summary, sort_keys=True)}", flush=True)
    return dict(summary)


def _run_finetune_stage(
    args: argparse.Namespace,
    *,
    label: str,
    config: str,
    init_checkpoint: str,
    output_root: str,
    init_classifier: bool,
) -> str:
    started = time.time()
    command = [
        sys.executable,
        "research/scripts/run_campp_finetune.py",
        "--config",
        config,
        "--init-checkpoint",
        init_checkpoint,
        "--device",
        "cuda",
    ]
    if init_classifier:
        command.append("--init-classifier-from-checkpoint")
    _run(command, label=label)
    checkpoint = _latest_checkpoint(Path(output_root), since=started)
    print(f"[{args.run_id}] {label} checkpoint={checkpoint}", flush=True)
    return checkpoint


def _latest_checkpoint(output_root: Path, *, since: float) -> str:
    candidates = [
        checkpoint
        for checkpoint in output_root.glob("*/campp_encoder.pt")
        if checkpoint.stat().st_mtime >= since - 5.0
    ]
    if not candidates:
        raise FileNotFoundError(f"No new campp_encoder.pt under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).as_posix()


def _run_public_tail(args: argparse.Namespace, *, checkpoint: str) -> dict[str, Any]:
    output_dir = Path(args.tail_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "research/scripts/run_official_campp_tail.py",
        "--checkpoint-path",
        checkpoint,
        "--manifest-csv",
        args.public_manifest_csv,
        "--template-csv",
        args.template_csv,
        "--data-root",
        args.data_root,
        "--output-dir",
        output_dir.as_posix(),
        "--experiment-id",
        args.run_id,
        "--encoder-backend",
        "torch",
        "--device",
        "cuda",
        "--search-device",
        "cuda",
        "--batch-size",
        "512",
        "--search-batch-size",
        "2048",
        "--top-cache-k",
        "200",
        "--mode",
        "segment_mean",
        "--eval-chunk-seconds",
        "6.0",
        "--segment-count",
        "3",
        "--long-file-threshold-seconds",
        "6.0",
    ]
    frontend_pack = Path(args.frontend_pack_dir)
    if frontend_pack.is_dir():
        command.extend(["--frontend-pack-dir", frontend_pack.as_posix()])
    _run(command, label="public-c4-tail")
    summary_path = output_dir / f"{args.run_id}_summary.json"
    summary = json.loads(summary_path.read_text())
    c4_submission = Path(str(summary["c4_submission_path"]))
    short_submission = output_dir / "submission.csv"
    shutil.copyfile(c4_submission, short_submission)
    summary["submission_csv"] = short_submission.as_posix()
    summary["submission_sha256"] = _sha256(short_submission)
    return dict(summary)


def _run(command: list[str], *, label: str) -> None:
    print(f"[ms37] {label} command: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
