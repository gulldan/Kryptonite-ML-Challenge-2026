from __future__ import annotations

import json
from pathlib import Path

from kryptonite.eval import (
    build_int8_feasibility_report,
    load_int8_feasibility_config,
    write_int8_feasibility_report,
)


def test_int8_feasibility_report_surfaces_no_go_blockers(tmp_path: Path) -> None:
    config_path = _write_fixture_config(tmp_path, structural_stub=True)

    config = load_int8_feasibility_config(config_path=config_path)
    report = build_int8_feasibility_report(config, config_path=config_path, project_root=tmp_path)

    assert report.summary.decision == "no_go"
    assert report.calibration_set.selected_input_count == 4
    assert report.calibration_set.category_counts == {
        "baseline": 1,
        "corruption": 2,
        "extreme_duration": 1,
    }
    failed_details = {check.detail for check in report.checks if not check.passed}
    assert any("structural-stub" in detail for detail in failed_details)
    assert any("model.plan" in detail for detail in failed_details)
    assert any("parity report" in detail.lower() for detail in failed_details)


def test_int8_feasibility_report_returns_go_when_artifacts_and_metrics_pass(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_config(
        tmp_path,
        structural_stub=False,
        include_fp16_engine=True,
        include_int8_engine=True,
        include_parity_report=True,
        include_metrics=True,
    )

    config = load_int8_feasibility_config(config_path=config_path)
    report = build_int8_feasibility_report(config, config_path=config_path, project_root=tmp_path)

    assert report.summary.decision == "go"
    assert report.deltas.eer_delta == 0.005
    assert report.deltas.min_dcf_delta == 0.01
    assert report.deltas.latency_speedup_ratio == 1.5


def test_write_int8_feasibility_report_emits_json_markdown_and_config_copy(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_config(
        tmp_path,
        structural_stub=False,
        include_fp16_engine=True,
        include_int8_engine=True,
        include_parity_report=True,
        include_metrics=True,
    )

    config = load_int8_feasibility_config(config_path=config_path)
    report = build_int8_feasibility_report(config, config_path=config_path, project_root=tmp_path)
    written = write_int8_feasibility_report(report)

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")

    assert payload["summary"]["decision"] == "go"
    assert "## Decision Checks" in markdown
    assert written.source_config_copy_path is not None
    assert Path(written.source_config_copy_path).is_file()


def _write_fixture_config(
    tmp_path: Path,
    *,
    structural_stub: bool,
    include_fp16_engine: bool = False,
    include_int8_engine: bool = False,
    include_parity_report: bool = False,
    include_metrics: bool = False,
) -> Path:
    assets_root = tmp_path / "assets" / "int8"
    artifacts_root = tmp_path / "artifacts"
    release_root = artifacts_root / "release" / "current"
    assets_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "model-bundle").mkdir(parents=True, exist_ok=True)

    (assets_root / "calibration_catalog.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "scenario_id": "alpha_reference",
                        "category": "baseline",
                        "audio_path": "artifacts/inference-stress/inputs/alpha_reference.wav",
                        "duration_seconds": 1.0,
                        "notes": "baseline",
                    },
                    {
                        "scenario_id": "alpha_short",
                        "category": "extreme_duration",
                        "audio_path": "artifacts/inference-stress/inputs/alpha_short.wav",
                        "duration_seconds": 0.25,
                        "notes": "short",
                    },
                    {
                        "scenario_id": "alpha_noisy",
                        "category": "corruption",
                        "audio_path": "artifacts/inference-stress/inputs/alpha_noisy.wav",
                        "duration_seconds": 1.0,
                        "notes": "noise",
                    },
                    {
                        "scenario_id": "alpha_echo",
                        "category": "corruption",
                        "audio_path": "artifacts/inference-stress/inputs/alpha_echo.wav",
                        "duration_seconds": 1.25,
                        "notes": "echo",
                    },
                    {
                        "scenario_id": "alpha_silence",
                        "category": "corruption",
                        "audio_path": "artifacts/inference-stress/inputs/alpha_silence.wav",
                        "duration_seconds": 1.0,
                        "notes": "silence",
                    },
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (artifacts_root / "model-bundle" / "metadata.json").write_text(
        json.dumps(
            {
                "model_version": "campp-prod-v4",
                "structural_stub": structural_stub,
                "export_boundary": {"export_profile": "baseline"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (artifacts_root / "model-bundle" / "model.onnx").write_bytes(b"onnx")
    if include_fp16_engine:
        (artifacts_root / "model-bundle" / "model.plan").write_bytes(b"fp16")
    if include_int8_engine:
        (artifacts_root / "model-bundle" / "model.int8.plan").write_bytes(b"int8")
    if include_parity_report:
        release_root.mkdir(parents=True, exist_ok=True)
        (release_root / "onnx_parity_report.json").write_text("{}\n", encoding="utf-8")
    if include_metrics:
        _write_verification_report(
            release_root / "fp16" / "verification_eval_report.json",
            eer=0.08,
            min_dcf=0.2,
        )
        _write_verification_report(
            release_root / "int8" / "verification_eval_report.json",
            eer=0.085,
            min_dcf=0.21,
        )
        _write_stress_report(
            release_root / "fp16" / "inference_stress_report.json",
            latency=9.0,
            rss=512.0,
            cuda=256.0,
        )
        _write_stress_report(
            release_root / "int8" / "inference_stress_report.json",
            latency=6.0,
            rss=544.0,
            cuda=320.0,
        )

    config_path = tmp_path / "configs" / "release" / "int8-feasibility.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'title = "Test INT8 feasibility"',
                'report_id = "test-int8-feasibility"',
                'candidate_label = "fixture"',
                'summary = "Fixture summary"',
                'output_root = "artifacts/release-decisions/test-int8-feasibility"',
                "",
                "[artifacts]",
                'model_bundle_metadata_path = "artifacts/model-bundle/metadata.json"',
                'onnx_model_path = "artifacts/model-bundle/model.onnx"',
                'fp16_engine_path = "artifacts/model-bundle/model.plan"',
                'int8_engine_path = "artifacts/model-bundle/model.int8.plan"',
                'onnx_parity_report_path = "artifacts/release/current/onnx_parity_report.json"',
                (
                    "fp16_verification_report_path = "
                    '"artifacts/release/current/fp16/verification_eval_report.json"'
                ),
                (
                    "fp16_stress_report_path = "
                    '"artifacts/release/current/fp16/inference_stress_report.json"'
                ),
                (
                    "int8_verification_report_path = "
                    '"artifacts/release/current/int8/verification_eval_report.json"'
                ),
                (
                    "int8_stress_report_path = "
                    '"artifacts/release/current/int8/inference_stress_report.json"'
                ),
                "",
                "[calibration_set]",
                'source_catalog_path = "assets/int8/calibration_catalog.json"',
                'include_categories = ["baseline", "extreme_duration", "corruption"]',
                'exclude_scenarios = ["alpha_silence"]',
                "short_max_duration_seconds = 1.0",
                "mid_max_duration_seconds = 4.0",
                "",
                "[gates]",
                "require_non_stub_model = true",
                "require_fp16_engine = true",
                "require_onnx_parity_report = true",
                "require_int8_engine = true",
                "max_eer_delta = 0.01",
                "max_min_dcf_delta = 0.02",
                "min_latency_speedup_ratio = 1.15",
                "max_process_rss_delta_mib = 64.0",
                "max_cuda_allocated_delta_mib = 128.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_verification_report(path: Path, *, eer: float, min_dcf: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "metrics": {
                        "trial_count": 128,
                        "eer": eer,
                        "min_dcf": min_dcf,
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_stress_report(path: Path, *, latency: float, rss: float, cuda: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "batch_bursts": [
                    {"batch_size": 1, "status": "passed", "mean_ms_per_audio": latency},
                ],
                "memory": {
                    "peak_process_rss_mib": rss,
                    "peak_cuda_allocated_mib": cuda,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
