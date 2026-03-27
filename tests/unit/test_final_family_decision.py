from __future__ import annotations

import json
from pathlib import Path

from kryptonite.eval.final_family_decision import (
    build_final_family_decision,
    load_final_family_decision_config,
    write_final_family_decision,
)


def test_final_family_decision_writes_machine_readable_report(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    artifacts_root = tmp_path / "artifacts"
    docs_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    verification_root = artifacts_root / "baselines" / "campp" / "run-001"
    verification_root.mkdir(parents=True, exist_ok=True)
    verification_report = verification_root / "verification_eval_report.json"
    score_summary = verification_root / "score_summary.json"
    baseline_report = verification_root / "campp_baseline_report.md"
    checkpoint = verification_root / "campp_encoder.pt"
    docs_file = docs_root / "campp-stage3-training.md"

    verification_report.write_text(
        json.dumps(
            {
                "summary": {
                    "metrics": {"eer": 0.123, "min_dcf": 0.456, "trial_count": 42},
                    "score_statistics": {
                        "mean_positive_score": 0.9,
                        "mean_negative_score": 0.4,
                        "score_gap": 0.5,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    score_summary.write_text(
        json.dumps(
            {
                "trial_count": 42,
                "mean_positive_score": 0.9,
                "mean_negative_score": 0.4,
                "score_gap": 0.5,
            }
        ),
        encoding="utf-8",
    )
    baseline_report.write_text("# CAM++ baseline\n", encoding="utf-8")
    checkpoint.write_text("stub", encoding="utf-8")
    docs_file.write_text("# Stage-3\n", encoding="utf-8")

    config_path = tmp_path / "final-family-decision.toml"
    config_path.write_text(
        """
title = "Fixture Final Family Decision"
decision_id = "fixture-final-family"
accepted_at = "2026-03-27"
summary = "Fixture summary"
context = "Fixture context"
decision = "Use CAM++."
output_root = "artifacts/release-decisions/fixture"
selected_production_student = "campp"
selected_stretch_teacher = "teacher"
decision_drivers = ["Freeze one family", "Keep teacher off the critical path"]
next_step_issues = ["KVA-538", "KVA-539"]

[[option]]
family_id = "campp"
label = "CAM++"
role = "production_student"
status = "selected"
summary = "Selected student."
rationale = "Most mature path."
export_readiness = "ready_next"
verification_report_path = "artifacts/baselines/campp/run-001/verification_eval_report.json"
score_summary_path = "artifacts/baselines/campp/run-001/score_summary.json"
report_markdown_path = "artifacts/baselines/campp/run-001/campp_baseline_report.md"
checkpoint_path = "artifacts/baselines/campp/run-001/campp_encoder.pt"
evidence_paths = ["docs/campp-stage3-training.md"]

[[option]]
family_id = "teacher"
label = "Teacher"
role = "stretch_teacher"
status = "selected"
summary = "Stretch branch."
rationale = "Later branch."
export_readiness = "deferred"
evidence_paths = ["docs/campp-stage3-training.md"]

[[option]]
family_id = "other"
label = "Other baseline"
role = "production_student"
status = "rejected"
summary = "Rejected baseline."
rationale = "Not ready."
export_readiness = "not_ready"
evidence_paths = ["docs/campp-stage3-training.md"]
rejected_reason = "Missing staged handoff."
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_final_family_decision_config(config_path=config_path)
    report = build_final_family_decision(config, project_root=tmp_path)
    written = write_final_family_decision(report, project_root=tmp_path)

    assert report.selected_production_student.family_id == "campp"
    assert report.selected_production_student.metrics.eer == 0.123
    assert report.selected_production_student.metrics.score_gap == 0.5
    assert report.selected_stretch_teacher.family_id == "teacher"
    assert report.rejected_options[0].family_id == "other"
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    assert payload["selected_production_student"]["family_id"] == "campp"
    assert payload["selected_stretch_teacher"]["family_id"] == "teacher"

    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")
    assert "## Rejected Alternatives" in markdown
    assert "CAM++" in markdown
    assert "Other baseline" in markdown


def test_checked_in_final_family_decision_config_builds_against_repository() -> None:
    config_path = Path("configs/release/final-family-decision.toml")

    config = load_final_family_decision_config(config_path=config_path)
    report = build_final_family_decision(config)

    assert report.decision_id == "kryptonite-2026-final-family"
    assert report.selected_production_student.family_id == "campp"
    assert report.selected_stretch_teacher.family_id == "wavlm_w2vbert_peft"
    assert any(option.family_id == "eres2netv2" for option in report.rejected_options)
    assert report.selected_production_student.metrics.score_gap is not None
