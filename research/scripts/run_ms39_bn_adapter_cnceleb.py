"""Orchestrate MS39 BN-affine CAM++ CN-Celeb adaptation branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BranchSpec:
    name: str
    run_id: str
    config: str
    init_checkpoint: str
    output_root: str
    tail_output_dir: str


def main() -> None:
    args = _parse_args()
    started_at = time.time()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(
        f"[ms39] start {datetime.now(UTC).isoformat()} gpu={gpu} "
        f"encoder_scope={args.encoder_trainable_scope}",
        flush=True,
    )
    _wait_for_cnceleb(args)
    manifest_summary = _build_cnceleb_manifest(args)
    branches = (
        BranchSpec(
            name="ms31",
            run_id=args.ms31_run_id,
            config=args.ms31_config,
            init_checkpoint=args.ms31_checkpoint,
            output_root=args.ms31_output_root,
            tail_output_dir=args.ms31_tail_output_dir,
        ),
        BranchSpec(
            name="ms32",
            run_id=args.ms32_run_id,
            config=args.ms32_config,
            init_checkpoint=args.ms32_checkpoint,
            output_root=args.ms32_output_root,
            tail_output_dir=args.ms32_tail_output_dir,
        ),
    )
    branch_summaries = []
    for branch in branches:
        branch_summaries.append(_run_branch(args, branch=branch))

    run_summary = {
        "started_at_utc": datetime.fromtimestamp(started_at, UTC).isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "gpu": gpu,
        "encoder_trainable_scope": args.encoder_trainable_scope,
        "manifest_summary": manifest_summary,
        "branches": branch_summaries,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms31-run-id", required=True)
    parser.add_argument("--ms32-run-id", required=True)
    parser.add_argument("--encoder-trainable-scope", default="batchnorm-affine")
    parser.add_argument("--cnceleb-root", default="datasets/CN-Celeb_flac")
    parser.add_argument("--wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--wait-poll-seconds", type=int, default=600)
    parser.add_argument("--manifest-output-dir", default="artifacts/manifests/cnceleb_v2_ms39")
    parser.add_argument("--manifest-experiment-id", default="cnceleb_v2_ms39")
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
        "--ms32-checkpoint",
        default=(
            "artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/"
            "20260414T055357Z-f1f2fa87143a/campp_encoder.pt"
        ),
    )
    parser.add_argument(
        "--ms31-config",
        default="research/configs/training/campp-ms39-ms31-bn-adapter-cnceleb.toml",
    )
    parser.add_argument(
        "--ms32-config",
        default="research/configs/training/campp-ms39b-ms32-bn-adapter-cnceleb.toml",
    )
    parser.add_argument(
        "--ms31-output-root",
        default="artifacts/baselines/campp-ms39-ms31-bn-adapter-cnceleb",
    )
    parser.add_argument(
        "--ms32-output-root",
        default="artifacts/baselines/campp-ms39b-ms32-bn-adapter-cnceleb",
    )
    parser.add_argument("--ms31-tail-output-dir", required=True)
    parser.add_argument("--ms32-tail-output-dir", required=True)
    parser.add_argument(
        "--public-manifest-csv",
        default="artifacts/eda/participants_public_baseline/test_public_manifest.csv",
    )
    parser.add_argument("--template-csv", default="datasets/Для участников/test_public.csv")
    parser.add_argument("--data-root", default="datasets/Для участников")
    parser.add_argument(
        "--frontend-pack-dir",
        default="artifacts/cache/campp-official-public-ms1-v1-pack",
    )
    parser.add_argument("--report-json", required=True)
    return parser.parse_args()


def _wait_for_cnceleb(args: argparse.Namespace) -> None:
    root = Path(args.cnceleb_root)
    deadline = time.time() + args.wait_timeout_seconds
    while True:
        if root.is_dir():
            print(f"[ms39] CN-Celeb root ready: {root}", flush=True)
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for {root}")
        archive = Path("datasets/cn-celeb_v2.tar.gz")
        archive_size = archive.stat().st_size if archive.exists() else 0
        print(
            f"[ms39] waiting for {root}; archive_size_gib={archive_size / 1024**3:.2f}",
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
    print(f"[ms39] manifest summary: {json.dumps(summary, sort_keys=True)}", flush=True)
    return dict(summary)


def _run_branch(args: argparse.Namespace, *, branch: BranchSpec) -> dict[str, Any]:
    started = time.time()
    train_command = [
        sys.executable,
        "research/scripts/run_campp_finetune.py",
        "--config",
        branch.config,
        "--init-checkpoint",
        branch.init_checkpoint,
        "--device",
        "cuda",
        "--encoder-trainable-scope",
        args.encoder_trainable_scope,
        "--output",
        "json",
    ]
    _run(train_command, label=f"{branch.name}-train")
    checkpoint = _latest_checkpoint(Path(branch.output_root), since=started)
    print(f"[ms39] {branch.name} checkpoint={checkpoint}", flush=True)
    tail_summary = _run_public_tail(args, branch=branch, checkpoint=checkpoint)
    return {
        "name": branch.name,
        "run_id": branch.run_id,
        "config": branch.config,
        "init_checkpoint": branch.init_checkpoint,
        "checkpoint": checkpoint,
        "output_root": branch.output_root,
        "tail_summary": tail_summary,
    }


def _latest_checkpoint(output_root: Path, *, since: float) -> str:
    candidates = [
        checkpoint
        for checkpoint in output_root.glob("*/campp_encoder.pt")
        if checkpoint.stat().st_mtime >= since - 5.0
    ]
    if not candidates:
        raise FileNotFoundError(f"No new campp_encoder.pt under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).as_posix()


def _run_public_tail(
    args: argparse.Namespace,
    *,
    branch: BranchSpec,
    checkpoint: str,
) -> dict[str, Any]:
    output_dir = Path(branch.tail_output_dir)
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
        branch.run_id,
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
    _run(command, label=f"{branch.name}-public-c4-tail")
    summary_path = output_dir / f"{branch.run_id}_summary.json"
    summary = json.loads(summary_path.read_text())
    c4_submission = Path(str(summary["c4_submission_path"]))
    short_submission = Path("artifacts/submissions") / f"{branch.run_id}_submission.csv"
    short_submission.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(c4_submission, short_submission)
    summary["submission_csv"] = short_submission.as_posix()
    summary["submission_sha256"] = _sha256(short_submission)
    return dict(summary)


def _run(command: list[str], *, label: str) -> None:
    print(f"[ms39] {label} command: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
