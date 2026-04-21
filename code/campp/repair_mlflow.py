#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from common import (
    load_config,
    resolve_mlflow_experiment,
    resolve_mlflow_tracking_uri,
    runs_root,
    submissions_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and optionally close stale MLflow runs.")
    parser.add_argument("--config", required=True, help="Path to CAM++ YAML config.")
    parser.add_argument(
        "--older-than-minutes",
        type=float,
        default=20.0,
        help="Only consider RUNNING runs older than this.",
    )
    parser.add_argument(
        "--status",
        default="auto",
        choices=["auto", "KILLED", "FAILED", "FINISHED"],
        help="Status to set when closing stale runs.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually terminate stale runs.")
    return parser.parse_args()


def collect_process_args() -> str:
    result = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, check=True)
    return result.stdout


def resolve_local_run_dir(config: dict, run_name: str) -> Path | None:
    for root in (runs_root(config), submissions_root(config)):
        candidate = root / run_name
        if candidate.exists():
            return candidate
    return None


def infer_status(run_dir: Path | None, default_status: str) -> str:
    if default_status != "auto":
        return default_status
    if run_dir is not None and (run_dir / "run_summary.json").exists():
        return "FINISHED"
    return "KILLED"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    from mlflow.tracking import MlflowClient

    tracking_uri = resolve_mlflow_tracking_uri(config)
    experiment_name = resolve_mlflow_experiment(config)
    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string='attributes.status = "RUNNING"',
        max_results=5000,
        order_by=["attributes.start_time DESC"],
    )
    process_args = collect_process_args()
    now_ms = int(time.time() * 1000)

    stale_runs: list[tuple[str, str, float, Path | None, str]] = []
    for run in runs:
        run_name = run.info.run_name or run.info.run_id
        age_minutes = max(0.0, (now_ms - int(run.info.start_time or now_ms)) / 60000.0)
        if age_minutes < float(args.older_than_minutes):
            continue
        if run_name and run_name in process_args:
            continue
        run_dir = resolve_local_run_dir(config, run_name)
        target_status = infer_status(run_dir, args.status)
        stale_runs.append((run.info.run_id, run_name, age_minutes, run_dir, target_status))

    if not stale_runs:
        print("No stale MLflow runs found.")
        return

    print("Stale MLflow runs:")
    for run_id, run_name, age_minutes, run_dir, target_status in stale_runs:
        print(
            f"- run_name={run_name} run_id={run_id} age_min={age_minutes:.1f} "
            f"status_if_closed={target_status} local_dir={run_dir or 'n/a'}"
        )

    if not args.apply:
        print("\nDry run only. Re-run with --apply to terminate these runs.")
        return

    for run_id, run_name, _age_minutes, _run_dir, target_status in stale_runs:
        client.set_terminated(run_id, status=target_status)
        print(f"Closed run {run_name} ({run_id}) with status={target_status}")


if __name__ == "__main__":
    main()
