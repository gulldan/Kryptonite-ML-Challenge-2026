from __future__ import annotations

import json
from pathlib import Path

import kryptonite.eval.teacher_student_robust_dev as robust_dev_module
from kryptonite.eval import (
    TeacherStudentRobustDevCandidateConfig,
    TeacherStudentRobustDevConfig,
    TeacherStudentRobustDevCorruptedSuitesConfig,
    TeacherStudentRobustDevSelectionConfig,
    build_teacher_student_robust_dev_report,
    render_teacher_student_robust_dev_markdown,
    write_teacher_student_robust_dev_report,
)
from kryptonite.eval.teacher_student_robust_dev_models import (
    CandidateEvidence,
    CorruptedSuiteEntry,
    TeacherStudentRobustDevCostSummary,
    TeacherStudentRobustDevSuiteEvaluation,
)


def test_teacher_student_robust_dev_ranks_candidates_and_writes_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "teacher-student-robust-dev.toml"
    config_path.write_text('title = "fixture"\n', encoding="utf-8")
    config = TeacherStudentRobustDevConfig(
        title="Teacher vs student robust dev",
        ticket_id="KVA-532",
        report_id="teacher-student-robust-dev",
        output_root="artifacts/eval/teacher-student-robust-dev",
        device="cpu",
        selection=TeacherStudentRobustDevSelectionConfig(
            clean_weight=0.25,
            corrupted_weight=0.75,
            eer_weight=0.7,
            min_dcf_weight=0.3,
        ),
        corrupted_suites=TeacherStudentRobustDevCorruptedSuitesConfig(
            catalog_path="artifacts/eval/corrupted-dev-suites/catalog.json",
            suite_ids=("dev_snr", "dev_reverb"),
            run_clean_dev=True,
        ),
        candidates=(
            TeacherStudentRobustDevCandidateConfig(
                candidate_id="teacher",
                label="Teacher",
                role="teacher",
                family="teacher_peft",
                run_root="artifacts/baselines/teacher",
            ),
            TeacherStudentRobustDevCandidateConfig(
                candidate_id="campp_student",
                label="CAM++ student",
                role="student",
                family="campp",
                run_root="artifacts/baselines/campp",
            ),
            TeacherStudentRobustDevCandidateConfig(
                candidate_id="eres_student",
                label="ERes student",
                role="student",
                family="eres2netv2",
                run_root="artifacts/baselines/eres",
            ),
        ),
        notes=("robust-dev fixture",),
    )
    catalog_suites = (
        CorruptedSuiteEntry(
            suite_id="dev_snr",
            family="noise",
            description="Noise suite",
            manifest_path="artifacts/eval/dev_snr/dev_manifest.jsonl",
            trial_manifest_paths=("artifacts/eval/dev_snr/trials.jsonl",),
        ),
        CorruptedSuiteEntry(
            suite_id="dev_reverb",
            family="reverb",
            description="Reverb suite",
            manifest_path="artifacts/eval/dev_reverb/dev_manifest.jsonl",
            trial_manifest_paths=("artifacts/eval/dev_reverb/trials.jsonl",),
        ),
    )
    evidence_by_candidate = {
        "teacher": _candidate_evidence(
            tmp_path,
            candidate_id="teacher",
            clean_eer=0.10,
            clean_min_dcf=0.20,
        ),
        "campp_student": _candidate_evidence(
            tmp_path,
            candidate_id="campp_student",
            clean_eer=0.08,
            clean_min_dcf=0.17,
        ),
        "eres_student": _candidate_evidence(
            tmp_path,
            candidate_id="eres_student",
            clean_eer=0.12,
            clean_min_dcf=0.23,
        ),
    }
    suite_results_by_candidate = {
        "teacher": (
            _suite_evaluation(
                suite_id="dev_snr",
                family="noise",
                eer=0.15,
                min_dcf=0.24,
            ),
            _suite_evaluation(
                suite_id="dev_reverb",
                family="reverb",
                eer=0.12,
                min_dcf=0.22,
            ),
        ),
        "campp_student": (
            _suite_evaluation(
                suite_id="dev_snr",
                family="noise",
                eer=0.11,
                min_dcf=0.19,
            ),
            _suite_evaluation(
                suite_id="dev_reverb",
                family="reverb",
                eer=0.10,
                min_dcf=0.18,
            ),
        ),
        "eres_student": (
            _suite_evaluation(
                suite_id="dev_snr",
                family="noise",
                eer=0.17,
                min_dcf=0.28,
            ),
            _suite_evaluation(
                suite_id="dev_reverb",
                family="reverb",
                eer=0.15,
                min_dcf=0.25,
            ),
        ),
    }

    def fake_load_corrupted_suites(
        *,
        project_root: Path,
        catalog_path: str,
        suite_ids: tuple[str, ...],
    ) -> tuple[CorruptedSuiteEntry, ...]:
        assert project_root == tmp_path
        assert catalog_path == config.corrupted_suites.catalog_path
        assert suite_ids == config.corrupted_suites.suite_ids
        return catalog_suites

    def fake_evaluate_candidate(
        *,
        candidate: TeacherStudentRobustDevCandidateConfig,
        config: TeacherStudentRobustDevConfig,
        suites: tuple[CorruptedSuiteEntry, ...],
        project_root: Path,
    ) -> tuple[
        TeacherStudentRobustDevCandidateConfig,
        CandidateEvidence,
        tuple[TeacherStudentRobustDevSuiteEvaluation, ...],
    ]:
        assert project_root == tmp_path
        assert suites == catalog_suites
        assert config.ticket_id == "KVA-532"
        return (
            candidate,
            evidence_by_candidate[candidate.candidate_id],
            suite_results_by_candidate[candidate.candidate_id],
        )

    monkeypatch.setattr(
        robust_dev_module,
        "load_corrupted_suites",
        fake_load_corrupted_suites,
    )
    monkeypatch.setattr(
        robust_dev_module,
        "evaluate_candidate",
        fake_evaluate_candidate,
    )

    report = build_teacher_student_robust_dev_report(
        config,
        config_path=config_path,
        project_root=tmp_path,
    )
    written = write_teacher_student_robust_dev_report(report, project_root=tmp_path)

    assert [candidate.candidate_id for candidate in report.candidates] == [
        "campp_student",
        "teacher",
        "eres_student",
    ]
    assert [candidate.rank for candidate in report.candidates] == [1, 2, 3]
    assert report.summary.teacher_candidate_id == "teacher"
    assert report.summary.best_quality_candidate_id == "campp_student"
    assert report.summary.corrupted_suite_ids == ("dev_snr", "dev_reverb")
    assert report.pairwise[0].student_candidate_id == "campp_student"
    assert report.pairwise[0].robust_eer_delta < 0.0
    assert report.pairwise[1].student_candidate_id == "eres_student"
    assert report.pairwise[1].robust_eer_delta > 0.0

    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()
    markdown = render_teacher_student_robust_dev_markdown(report)
    assert "Quality Leaderboard" in markdown
    assert "Teacher Vs Students" in markdown
    assert "robust-dev fixture" in markdown

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    assert payload["summary"]["best_quality_candidate_id"] == "campp_student"
    assert payload["pairwise"][1]["student_candidate_id"] == "eres_student"


def _candidate_evidence(
    tmp_path: Path,
    *,
    candidate_id: str,
    clean_eer: float,
    clean_min_dcf: float,
) -> CandidateEvidence:
    return CandidateEvidence(
        run_root=tmp_path / "artifacts" / "baselines" / candidate_id,
        clean_report_markdown_path=f"artifacts/baselines/{candidate_id}/verification_eval_report.md",
        clean_trial_count=128,
        clean_eer=clean_eer,
        clean_min_dcf=clean_min_dcf,
        clean_score_gap=0.42,
        cost=TeacherStudentRobustDevCostSummary(
            training_device="cpu",
            precision="fp32",
            train_batch_size=2,
            eval_batch_size=2,
            gradient_accumulation_steps=1,
            effective_batch_size=2,
            max_epochs=1,
            train_row_count=32,
            dev_row_count=8,
            total_parameters=1_000_000,
            trainable_parameters=250_000,
            checkpoint_size_bytes=1_048_576,
            embedding_dim=192,
        ),
        train_manifest_path="artifacts/manifests/train.jsonl",
        max_dev_rows=None,
    )


def _suite_evaluation(
    *,
    suite_id: str,
    family: str,
    eer: float,
    min_dcf: float,
) -> TeacherStudentRobustDevSuiteEvaluation:
    return TeacherStudentRobustDevSuiteEvaluation(
        suite_id=suite_id,
        family=family,
        manifest_path=f"artifacts/eval/{suite_id}/dev_manifest.jsonl",
        trials_path=f"artifacts/eval/{suite_id}/trials.jsonl",
        output_root=f"artifacts/eval/candidates/example/robust_dev/{suite_id}",
        report_markdown_path=f"artifacts/eval/{suite_id}/verification_eval_report.md",
        trial_count=64,
        eer=eer,
        min_dcf=min_dcf,
        score_gap=0.25,
    )
