from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.serve import (
    build_inference_stress_report,
    generate_inference_stress_inputs,
    render_inference_stress_markdown,
    write_inference_stress_report,
)


def test_inference_stress_report_covers_audio_bursts_and_malformed_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.artifacts_root="artifacts"',
            'paths.manifests_root="artifacts/manifests"',
            'deployment.model_bundle_root="artifacts/model-bundle"',
            'deployment.demo_subset_root="artifacts/demo-subset"',
            'deployment.enrollment_cache_root="artifacts/enrollment-cache"',
        ],
    )
    _patch_runtime_probes(monkeypatch)

    report = build_inference_stress_report(
        config=config,
        batch_sizes=(1, 4),
        benchmark_iterations=1,
        warmup_iterations=0,
    )
    written = write_inference_stress_report(report, output_root=tmp_path / "artifacts" / "reports")

    scenario_ids = {result.scenario_id for result in report.audio_scenarios}
    malformed_ids = {result.scenario_id for result in report.malformed_requests}

    assert report.summary.passed is True
    assert report.service.requested_backend == "auto"
    assert report.service.selected_backend == "torch"
    assert report.service.implementation == "feature_statistics"
    assert report.summary.control_ordering_passed is True
    assert report.summary.long_audio_chunking_observed is True
    assert report.memory.peak_process_rss_mib is not None
    assert report.memory.peak_process_rss_delta_mib is not None
    assert scenario_ids == {
        "alpha_reference",
        "bravo_reference",
        "alpha_short",
        "alpha_long_bursty",
        "alpha_noisy",
        "alpha_echo",
        "alpha_clipped",
        "alpha_silence",
    }
    assert [result.batch_size for result in report.batch_bursts] == [1, 4]
    assert all(result.status == "passed" for result in report.batch_bursts)
    assert malformed_ids == {
        "missing_audio_path",
        "corrupt_audio",
        "invalid_stage",
        "invalid_schema",
    }
    assert all(result.matched_expectation for result in report.malformed_requests)
    assert report.hard_limits.largest_validated_batch_size == 4
    assert report.hard_limits.max_validated_duration_seconds is not None
    assert report.hard_limits.max_validated_duration_seconds > 4.0
    assert all(result.memory is not None for result in report.audio_scenarios)
    assert all(result.memory is not None for result in report.batch_bursts)
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()
    assert "Hard Limits" in render_inference_stress_markdown(report)


def test_generate_inference_stress_inputs_writes_catalog_and_assets(tmp_path: Path) -> None:
    generated = generate_inference_stress_inputs(project_root=tmp_path, artifacts_root="artifacts")

    catalog_path = tmp_path / generated.catalog_path
    input_root = tmp_path / generated.input_root

    assert catalog_path.is_file()
    assert input_root.is_dir()
    assert len(generated.inputs) >= 10
    assert any(descriptor.scenario_id == "alpha_long_bursty" for descriptor in generated.inputs)
    assert any(descriptor.scenario_id == "corrupt_audio" for descriptor in generated.inputs)


def _patch_runtime_probes(monkeypatch) -> None:
    def fake_load_module(module_name: str) -> object:
        if module_name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: False),
                version=SimpleNamespace(cuda=None),
            )
        if module_name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: "1.0.0")
