from __future__ import annotations

import json
from pathlib import Path

import kryptonite.eval.teacher_student_robust_dev_runtime as runtime_module
from kryptonite.config import ChunkingConfig, FeaturesConfig, NormalizationConfig, VADConfig
from kryptonite.eval import (
    TeacherStudentRobustDevCandidateConfig,
    TeacherStudentRobustDevConfig,
    TeacherStudentRobustDevCorruptedSuitesConfig,
    TeacherStudentRobustDevSelectionConfig,
    TeacherStudentRobustDevSuiteEvaluation,
)
from kryptonite.eval.teacher_student_robust_dev_models import CorruptedSuiteEntry
from kryptonite.eval.teacher_student_robust_dev_runtime import (
    CostFields,
    RuntimeConfig,
    evaluate_candidate,
    load_corrupted_suites,
    resolve_suite_trials,
)
from kryptonite.training.speaker_baseline import TRAINING_SUMMARY_FILE_NAME


def test_evaluate_candidate_resolves_output_root_from_project_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "artifacts" / "baselines" / "teacher"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / TRAINING_SUMMARY_FILE_NAME).write_text(
        json.dumps(
            {
                "device": "cuda",
                "train_row_count": 42,
                "dev_row_count": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "verification_eval_report.json").write_text(
        json.dumps(
            {
                "summary": {
                    "metrics": {
                        "trial_count": 12,
                        "eer": 0.11,
                        "min_dcf": 0.21,
                    },
                    "score_statistics": {
                        "score_gap": 0.33,
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "verification_eval_report.md").write_text("# clean\n", encoding="utf-8")

    candidate = TeacherStudentRobustDevCandidateConfig(
        candidate_id="teacher",
        label="Teacher",
        role="teacher",
        family="teacher_peft",
        run_root="artifacts/baselines/teacher",
    )
    student_candidate = TeacherStudentRobustDevCandidateConfig(
        candidate_id="campp_student",
        label="CAM++ student",
        role="student",
        family="campp",
        run_root="artifacts/baselines/campp",
    )
    config = TeacherStudentRobustDevConfig(
        title="fixture",
        ticket_id="KVA-532",
        report_id="robust-dev",
        output_root="artifacts/eval/teacher-student-robust-dev",
        device="cpu",
        selection=TeacherStudentRobustDevSelectionConfig(),
        corrupted_suites=TeacherStudentRobustDevCorruptedSuitesConfig(
            catalog_path="artifacts/eval/corrupted-dev-suites/catalog.json",
            suite_ids=("dev_snr",),
            run_clean_dev=True,
        ),
        candidates=(candidate, student_candidate),
    )
    runtime = RuntimeConfig(
        normalization=_normalization_config(),
        vad=_vad_config(),
        features=_features_config(),
        chunking=_chunking_config(),
        train_manifest_path="artifacts/manifests/train.jsonl",
        max_dev_rows=None,
        eval_batch_size=2,
        device="cpu",
    )
    baseline_config: dict[str, object] = {
        "project": {
            "runtime": {"device": "cpu"},
            "training": {
                "precision": "fp32",
                "batch_size": 2,
                "eval_batch_size": 2,
                "max_epochs": 1,
            },
        },
        "optimization": {
            "gradient_accumulation_steps": 1,
        },
    }
    captured: dict[str, Path] = {}

    def fake_load_candidate_artifacts(
        *,
        candidate: TeacherStudentRobustDevCandidateConfig,
        run_root: Path,
        project_root: Path,
    ) -> tuple[dict[str, object], RuntimeConfig, CostFields]:
        assert candidate.candidate_id == "teacher"
        assert run_root == tmp_path / "artifacts" / "baselines" / "teacher"
        assert project_root == tmp_path
        return baseline_config, runtime, CostFields(123, 45, 256, 2048)

    def fake_evaluate_candidate_suites(
        *,
        candidate: TeacherStudentRobustDevCandidateConfig,
        runtime: RuntimeConfig,
        suites: tuple[CorruptedSuiteEntry, ...],
        config: TeacherStudentRobustDevConfig,
        project_root: Path,
        report_output_root: Path,
    ) -> tuple[tuple[TeacherStudentRobustDevSuiteEvaluation, ...], CostFields]:
        del candidate, runtime, suites, config, project_root
        captured["report_output_root"] = report_output_root
        return (
            (
                TeacherStudentRobustDevSuiteEvaluation(
                    suite_id="dev_snr",
                    family="noise",
                    manifest_path="artifacts/eval/dev_snr/dev_manifest.jsonl",
                    trials_path="artifacts/eval/dev_snr/trials.jsonl",
                    output_root="artifacts/eval/dev_snr",
                    report_markdown_path="artifacts/eval/dev_snr/verification_eval_report.md",
                    trial_count=8,
                    eer=0.12,
                    min_dcf=0.24,
                    score_gap=0.3,
                ),
            ),
            CostFields(456, 78, 256, 2048),
        )

    monkeypatch.setattr(
        runtime_module,
        "load_candidate_artifacts",
        fake_load_candidate_artifacts,
    )
    monkeypatch.setattr(
        runtime_module,
        "evaluate_candidate_suites",
        fake_evaluate_candidate_suites,
    )

    returned_candidate, evidence, suite_results = evaluate_candidate(
        candidate=candidate,
        config=config,
        suites=(
            CorruptedSuiteEntry(
                suite_id="dev_snr",
                family="noise",
                description="Noise suite",
                manifest_path="artifacts/eval/dev_snr/dev_manifest.jsonl",
                trial_manifest_paths=("artifacts/eval/dev_snr/trials.jsonl",),
            ),
        ),
        project_root=tmp_path,
    )

    assert returned_candidate.candidate_id == "teacher"
    assert suite_results[0].suite_id == "dev_snr"
    assert captured["report_output_root"] == (
        tmp_path / "artifacts" / "eval" / "teacher-student-robust-dev" / "candidates" / "teacher"
    )
    assert evidence.run_root == run_root
    assert evidence.clean_report_markdown_path == (
        "artifacts/baselines/teacher/verification_eval_report.md"
    )
    assert evidence.cost.total_parameters == 456
    assert evidence.cost.trainable_parameters == 78


def test_load_corrupted_suites_filters_requested_order(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "suite_id": "dev_snr",
                        "family": "noise",
                        "description": "Noise suite",
                        "manifest_path": "artifacts/eval/dev_snr/dev_manifest.jsonl",
                        "trial_manifest_paths": ["artifacts/eval/dev_snr/trials.jsonl"],
                    },
                    {
                        "suite_id": "dev_reverb",
                        "family": "reverb",
                        "description": "Reverb suite",
                        "manifest_path": "artifacts/eval/dev_reverb/dev_manifest.jsonl",
                        "trial_manifest_paths": ["artifacts/eval/dev_reverb/trials.jsonl"],
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    suites = load_corrupted_suites(
        project_root=tmp_path,
        catalog_path="catalog.json",
        suite_ids=("dev_reverb", "dev_snr"),
    )

    assert [suite.suite_id for suite in suites] == ["dev_reverb", "dev_snr"]


def test_resolve_suite_trials_merges_and_deduplicates_rows(tmp_path: Path) -> None:
    first_trials_path = tmp_path / "first.jsonl"
    second_trials_path = tmp_path / "second.jsonl"
    first_trials_path.write_text(
        "".join(
            [
                json.dumps(
                    {
                        "left_id": "speaker_a:a",
                        "right_id": "speaker_a:b",
                        "label": 1,
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "left_id": "speaker_a:a",
                        "right_id": "speaker_b:a",
                        "label": 0,
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )
    second_trials_path.write_text(
        "".join(
            [
                json.dumps(
                    {
                        "left_id": "speaker_a:a",
                        "right_id": "speaker_b:a",
                        "label": 0,
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "left_id": "speaker_c:a",
                        "right_id": "speaker_c:b",
                        "label": 1,
                    }
                )
                + "\n",
            ]
        ),
        encoding="utf-8",
    )

    trials_path, trial_rows = resolve_suite_trials(
        suite=CorruptedSuiteEntry(
            suite_id="dev_snr",
            family="noise",
            description="Noise suite",
            manifest_path="artifacts/eval/dev_snr/dev_manifest.jsonl",
            trial_manifest_paths=("first.jsonl", "second.jsonl"),
        ),
        output_root=tmp_path / "output",
        metadata_rows=[],
        project_root=tmp_path,
    )

    assert trials_path.is_file()
    assert len(trial_rows) == 3
    assert trial_rows[0]["label"] == 1
    assert trial_rows[-1]["right_id"] == "speaker_c:b"


def _normalization_config() -> NormalizationConfig:
    return NormalizationConfig(
        target_sample_rate_hz=16_000,
        target_channels=1,
        output_format="wav",
        output_pcm_bits_per_sample=16,
        peak_headroom_db=1.0,
        dc_offset_threshold=0.01,
        clipped_sample_threshold=0.01,
    )


def _vad_config() -> VADConfig:
    return VADConfig(mode="none")


def _features_config() -> FeaturesConfig:
    return FeaturesConfig(
        sample_rate_hz=16_000,
        num_mel_bins=80,
        frame_length_ms=25.0,
        frame_shift_ms=10.0,
        fft_size=512,
        window_type="hann",
        f_min_hz=20.0,
    )


def _chunking_config() -> ChunkingConfig:
    return ChunkingConfig()
