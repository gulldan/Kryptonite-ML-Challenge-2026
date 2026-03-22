from __future__ import annotations

import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.tracking import build_tracker, create_run_id


def test_create_run_id_is_unique() -> None:
    first = create_run_id()
    second = create_run_id()

    assert first != second


def test_local_tracker_writes_run_files(tmp_path: Path) -> None:
    config = load_project_config(
        config_path=Path("configs/base.toml"),
        overrides=[
            f'tracking.output_root="{tmp_path.as_posix()}"',
            'tracking.run_name_prefix="unit"',
        ],
    )
    tracker = build_tracker(config=config)

    run = tracker.start_run(kind="train", config=config.to_dict(mask_secrets=True))
    artifact_source = tmp_path / "artifact.txt"
    artifact_source.write_text("ok\n")
    run.log_metrics({"loss": 0.1}, step=1)
    artifact_entry = run.log_artifact(artifact_source)
    summary = run.finish(summary={"status": "ok"})

    assert run.run_dir.exists()
    assert json.loads((run.run_dir / "params.json").read_text())["tracking"]["backend"] == "local"
    assert "loss" in (run.run_dir / "metrics.jsonl").read_text()
    assert Path(artifact_entry["stored_path"]).exists()
    assert summary["status"] == "completed"
