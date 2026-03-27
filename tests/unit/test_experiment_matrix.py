from __future__ import annotations

import json
from pathlib import Path

from kryptonite.training.experiment_matrix import (
    build_experiment_matrix,
    load_experiment_matrix_config,
    write_experiment_matrix,
)


def test_experiment_matrix_writes_machine_readable_report(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    scripts_root = tmp_path / "scripts"
    configs_root = tmp_path / "configs" / "training"
    docs_root.mkdir(parents=True, exist_ok=True)
    scripts_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)

    (docs_root / "campp.md").write_text("# CAM++\n", encoding="utf-8")
    (docs_root / "teacher.md").write_text("# Teacher\n", encoding="utf-8")
    (scripts_root / "run_campp.py").write_text("print('campp')\n", encoding="utf-8")
    (configs_root / "campp.toml").write_text("value = 1\n", encoding="utf-8")

    config_path = tmp_path / "experiment-matrix.toml"
    config_path.write_text(
        """
title = "Fixture Experiment Matrix"
matrix_id = "fixture-experiment-matrix"
accepted_at = "2026-03-28"
summary = "Fixture summary"
context = "Fixture context"
output_root = "artifacts/planning/fixture-experiment-matrix"
assumptions = ["One GPU", "Teacher is deferred"]
validation_commands = [
  "uv run python scripts/build_experiment_matrix.py"
]

[[experiment]]
sequence = 1
experiment_id = "campp_stage1"
label = "CAM++ stage-1"
linear_issue = "KVA-519"
family = "CAM++"
track = "student_training"
priority = "P0"
repo_state = "ready_gpu"
summary = "Warm start."
expected_effect = "Create anchor checkpoint."
gpu_hours_low = 4.0
gpu_hours_high = 8.0
depends_on = []
evidence_paths = ["configs/training/campp.toml", "scripts/run_campp.py", "docs/campp.md"]
command = "uv run python scripts/run_campp.py"
notes = ["First real run."]

[[experiment]]
sequence = 2
experiment_id = "teacher_peft"
label = "Teacher PEFT"
linear_issue = "KVA-531"
family = "teacher"
track = "stretch_teacher"
priority = "P3"
repo_state = "deferred_stretch"
summary = "Stretch row."
expected_effect = "Potential upside."
gpu_hours_low = 12.0
gpu_hours_high = 20.0
depends_on = []
evidence_paths = ["docs/teacher.md"]
notes = ["Planning only."]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_experiment_matrix_config(config_path=config_path)
    report = build_experiment_matrix(config, project_root=tmp_path)
    written = write_experiment_matrix(report, project_root=tmp_path)

    assert report.ready_budget.gpu_hours_low == 4.0
    assert report.ready_budget.gpu_hours_high == 8.0
    assert report.deferred_budget.gpu_hours_low == 12.0
    assert report.deferred_budget.gpu_hours_high == 20.0
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    assert payload["matrix_id"] == "fixture-experiment-matrix"
    assert payload["priority_budgets"]["P0"]["gpu_hours_high"] == 8.0
    assert payload["priority_budgets"]["P3"]["gpu_hours_low"] == 12.0

    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")
    assert "## Priority Matrix" in markdown
    assert "CAM++ stage-1" in markdown
    assert "Teacher PEFT" in markdown
    assert "ready gpu" in markdown
    assert "deferred stretch" in markdown


def test_checked_in_experiment_matrix_config_builds_against_repository() -> None:
    config_path = Path("configs/training/experiment-matrix-v1.toml")

    config = load_experiment_matrix_config(config_path=config_path)
    report = build_experiment_matrix(config)

    experiment_ids = {experiment.experiment_id for experiment in report.experiments}
    assert report.matrix_id == "kryptonite-2026-experiment-matrix-v1"
    assert report.ready_budget.gpu_hours_low > 0
    assert "campp_stage1" in experiment_ids
    assert "eres2netv2_baseline" in experiment_ids
    assert "teacher_peft" in experiment_ids
    assert "distillation" in experiment_ids


def test_experiment_matrix_docs_are_linked_from_repository_indexes() -> None:
    root_readme = Path("README.md").read_text(encoding="utf-8")
    docs_readme = Path("docs/README.md").read_text(encoding="utf-8")
    doc_text = Path("docs/experiment-matrix-v1.md").read_text(encoding="utf-8")

    assert "docs/experiment-matrix-v1.md" in root_readme
    assert "docs/experiment-matrix-v1.md" in docs_readme
    assert "configs/training/experiment-matrix-v1.toml" in doc_text
    assert "scripts/build_experiment_matrix.py" in doc_text
    assert "KVA-531" in doc_text
