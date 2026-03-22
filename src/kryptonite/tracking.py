"""Lightweight experiment tracking with a local backend by default."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig


@dataclass(slots=True)
class LocalTrackingRun:
    run_id: str
    kind: str
    experiment: str
    run_name: str
    run_dir: Path
    copy_artifacts: bool

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        metrics_path = self.run_dir / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            payload = {"step": step, "metrics": metrics}
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    def log_artifact(self, source: Path) -> dict[str, Any]:
        source_path = source.resolve()
        artifact_dir = self.run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        target_path = artifact_dir / source_path.name

        if self.copy_artifacts:
            shutil.copy2(source_path, target_path)
        else:
            target_path = source_path

        payload = {
            "source": str(source_path),
            "stored_path": str(target_path),
            "copied": self.copy_artifacts,
        }
        artifacts_path = self.run_dir / "artifacts.json"
        entries = []
        if artifacts_path.exists():
            entries = json.loads(artifacts_path.read_text())
        entries.append(payload)
        artifacts_path.write_text(json.dumps(entries, indent=2, sort_keys=True))
        return payload

    def finish(self, *, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        run_metadata = json.loads((self.run_dir / "run.json").read_text())
        run_metadata["finished_at"] = utc_now()
        run_metadata["status"] = "completed"
        run_metadata["summary"] = summary or {}
        (self.run_dir / "run.json").write_text(
            json.dumps(run_metadata, indent=2, sort_keys=True)
        )
        return run_metadata


@dataclass(slots=True)
class LocalTracker:
    experiment: str
    output_root: Path
    run_name_prefix: str
    copy_artifacts: bool

    def start_run(self, *, kind: str, config: dict[str, Any]) -> LocalTrackingRun:
        run_id = create_run_id()
        run_name = f"{self.run_name_prefix}-{kind}-{run_id[:8]}"
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "params.json").write_text(json.dumps(config, indent=2, sort_keys=True))
        (run_dir / "metrics.jsonl").write_text("")
        (run_dir / "artifacts.json").write_text("[]")
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "kind": kind,
                    "experiment": self.experiment,
                    "run_name": run_name,
                    "started_at": utc_now(),
                    "status": "running",
                },
                indent=2,
                sort_keys=True,
            )
        )

        return LocalTrackingRun(
            run_id=run_id,
            kind=kind,
            experiment=self.experiment,
            run_name=run_name,
            run_dir=run_dir,
            copy_artifacts=self.copy_artifacts,
        )


def build_tracker(*, config: ProjectConfig) -> LocalTracker:
    if not config.tracking.enabled:
        raise RuntimeError("Tracking is disabled in the active config.")
    if config.tracking.backend != "local":
        raise RuntimeError(
            "Tracking backend "
            f"'{config.tracking.backend}' "
            "is not available without extra dependencies."
        )

    output_root = Path(config.tracking.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return LocalTracker(
        experiment=config.tracking.experiment,
        output_root=output_root,
        run_name_prefix=config.tracking.run_name_prefix,
        copy_artifacts=config.tracking.copy_artifacts,
    )


def create_run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
