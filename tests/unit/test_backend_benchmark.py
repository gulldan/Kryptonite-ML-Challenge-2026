from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import kryptonite.eval.backend_benchmark_builder as backend_benchmark_builder
import kryptonite.eval.backend_benchmark_runtime as backend_benchmark_runtime
from kryptonite.config import load_project_config
from kryptonite.eval import (
    BackendBenchmarkWorkloadResult,
    build_backend_benchmark_report,
    load_backend_benchmark_config,
    write_backend_benchmark_report,
)
from kryptonite.models import CAMPPlusConfig, CAMPPlusEncoder
from kryptonite.serve.onnx_export import (
    CAMPPONNXExportRequest,
    ONNXSmokeValidation,
    export_campp_checkpoint_to_onnx,
)


def test_backend_benchmark_report_writes_outputs_and_plots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = _write_backend_benchmark_fixture(tmp_path, monkeypatch=monkeypatch)
    config = load_backend_benchmark_config(config_path=config_path)
    fake_results = _build_fake_results(config=config)
    monkeypatch.setattr(
        backend_benchmark_builder,
        "run_backend_benchmark",
        lambda **_: fake_results,
    )

    report = build_backend_benchmark_report(config, config_path=config_path)
    written = write_backend_benchmark_report(report)

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")

    assert report.summary.passed is True
    assert {asset.batch_size for asset in report.plot_assets} == {1, 4}
    assert Path(written.workload_rows_path).is_file()
    assert all(Path(path).is_file() for path in written.plot_paths)
    assert payload["summary"]["passed"] is True
    assert len(payload["backend_summaries"]) == 3
    assert "Latency Graphs" in markdown
    assert "backend_benchmark_latency_batch1.svg" in markdown


def test_backend_benchmark_report_marks_failed_backend(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_backend_benchmark_fixture(tmp_path, monkeypatch=monkeypatch)
    config = load_backend_benchmark_config(config_path=config_path)
    fake_results = _build_fake_results(config=config, failed_backend="tensorrt")
    monkeypatch.setattr(
        backend_benchmark_builder,
        "run_backend_benchmark",
        lambda **_: fake_results,
    )

    report = build_backend_benchmark_report(config, config_path=config_path)
    written = write_backend_benchmark_report(report)
    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")

    assert report.summary.passed is False
    assert report.summary.successful_backend_count == 2
    assert "Failures" in markdown
    assert "`tensorrt`" in markdown


def test_load_backend_benchmark_config_requires_batch1_and_batched(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "release" / "backend-benchmark.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'title = "Fixture backend benchmark"',
                'report_id = "fixture-backend-benchmark"',
                'summary = "Fixture summary."',
                'project_root = "."',
                'output_root = "artifacts/release/current/backend-benchmark"',
                "",
                "[artifacts]",
                'model_bundle_metadata_path = "artifacts/model-bundle/metadata.json"',
                (
                    "tensorrt_report_path = "
                    '"artifacts/release/current/fp16/tensorrt_fp16_engine_report.json"'
                ),
                "",
                "[evaluation]",
                "seed = 7",
                'device = "auto"',
                'onnxruntime_provider = "auto"',
                "warmup_iterations = 1",
                "benchmark_iterations = 2",
                'backends = ["torch", "onnxruntime", "tensorrt"]',
                "max_mean_abs_diff = 0.01",
                "max_cosine_distance = 0.001",
                "",
                "[[workloads]]",
                'id = "single_only"',
                "batch_size = 1",
                "frame_count = 100",
                'description = "Only single workload"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_size>1"):
        load_backend_benchmark_config(config_path=config_path)


def test_backend_benchmark_artifacts_accept_single_profile_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = _write_backend_benchmark_fixture(
        tmp_path,
        monkeypatch=monkeypatch,
        use_single_profile=True,
    )
    config = load_backend_benchmark_config(config_path=config_path)

    artifacts = backend_benchmark_runtime.resolve_backend_benchmark_artifacts(config=config)

    assert len(artifacts.tensorrt_profiles) == 1
    profile = artifacts.tensorrt_profiles[0]
    assert profile.profile_id == "default"
    assert profile.min_shape == (1, 80, 16)
    assert profile.opt_shape == (4, 120, 16)
    assert profile.max_shape == (8, 640, 16)


def _write_backend_benchmark_fixture(
    tmp_path: Path,
    *,
    monkeypatch,
    use_single_profile: bool = False,
) -> Path:
    checkpoint_dir = tmp_path / "artifacts" / "baselines" / "campp" / "run-001"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "campp_encoder.pt"

    model_config = CAMPPlusConfig(
        feat_dim=16,
        embedding_size=32,
        growth_rate=4,
        bottleneck_scale=2,
        init_channels=8,
        head_channels=4,
        head_res_blocks=(1, 1),
        block_layers=(1, 1, 1),
        block_kernel_sizes=(3, 3, 3),
        block_dilations=(1, 1, 2),
        memory_efficient=False,
    )
    model = CAMPPlusEncoder(model_config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": {},
            "model_config": asdict(model_config),
            "baseline_config": {},
            "speaker_to_index": {"speaker_alpha": 0},
        },
        checkpoint_path,
    )

    project_config = load_project_config(
        config_path=Path("configs/base.toml"),
        overrides=[
            f'paths.project_root="{tmp_path.as_posix()}"',
            "features.num_mel_bins=16",
            "tracking.enabled=false",
            "runtime.num_workers=0",
        ],
    )
    monkeypatch.setattr(
        "kryptonite.serve.onnx_export._run_onnxruntime_smoke",
        lambda **_: ONNXSmokeValidation(
            checker_passed=True,
            onnxruntime_smoke_passed=True,
            sample_input_shape=(1, 120, 16),
            sample_output_shape=(1, 32),
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
        ),
    )
    exported = export_campp_checkpoint_to_onnx(
        config=project_config,
        request=CAMPPONNXExportRequest(
            checkpoint_path=str(checkpoint_dir),
            output_root="artifacts/model-bundle-campp-test",
            sample_frame_count=120,
        ),
    )

    engine_path = tmp_path / "artifacts" / "model-bundle-campp-test" / "model.plan"
    engine_path.write_bytes(b"fake-plan")
    tensorrt_report_path = (
        tmp_path / "artifacts" / "release" / "current" / "fp16" / "tensorrt_fp16_engine_report.json"
    )
    tensorrt_report_path.parent.mkdir(parents=True, exist_ok=True)
    tensorrt_report_path.write_text(
        json.dumps(
            _build_tensorrt_report_payload(
                engine_path="artifacts/model-bundle-campp-test/model.plan",
                source_checkpoint_path=exported.source_checkpoint_path,
                input_name=exported.input_name,
                output_name=exported.output_name,
                use_single_profile=use_single_profile,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "configs" / "release" / "backend-benchmark.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'title = "Fixture backend benchmark"',
                'report_id = "fixture-backend-benchmark"',
                'summary = "Fixture summary."',
                f'project_root = "{tmp_path.as_posix()}"',
                'output_root = "artifacts/release/current/backend-benchmark"',
                "",
                "[artifacts]",
                f'model_bundle_metadata_path = "{exported.metadata_path}"',
                (
                    "tensorrt_report_path = "
                    '"artifacts/release/current/fp16/tensorrt_fp16_engine_report.json"'
                ),
                "",
                "[evaluation]",
                "seed = 7",
                'device = "auto"',
                'onnxruntime_provider = "auto"',
                "warmup_iterations = 1",
                "benchmark_iterations = 2",
                'backends = ["torch", "onnxruntime", "tensorrt"]',
                "max_mean_abs_diff = 0.01",
                "max_cosine_distance = 0.001",
                "",
                "[[workloads]]",
                'id = "single_short"',
                "batch_size = 1",
                "frame_count = 100",
                'description = "single short"',
                "",
                "[[workloads]]",
                'id = "single_long"',
                "batch_size = 1",
                "frame_count = 240",
                'description = "single long"',
                "",
                "[[workloads]]",
                'id = "batch4_short"',
                "batch_size = 4",
                "frame_count = 100",
                'description = "batch4 short"',
                "",
                "[[workloads]]",
                'id = "batch4_long"',
                "batch_size = 4",
                "frame_count = 240",
                'description = "batch4 long"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _build_tensorrt_report_payload(
    *,
    engine_path: str,
    source_checkpoint_path: str,
    input_name: str,
    output_name: str,
    use_single_profile: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "engine_path": engine_path,
        "source_checkpoint_path": source_checkpoint_path,
        "input_name": input_name,
        "output_name": output_name,
    }
    if use_single_profile:
        payload["profile"] = {
            "min_shape": [1, 80, 16],
            "opt_shape": [4, 120, 16],
            "max_shape": [8, 640, 16],
        }
        return payload
    payload["profiles"] = [
        {
            "profile_id": "short",
            "min_shape": [1, 80, 16],
            "opt_shape": [4, 120, 16],
            "max_shape": [8, 160, 16],
        },
        {
            "profile_id": "long",
            "min_shape": [1, 160, 16],
            "opt_shape": [4, 320, 16],
            "max_shape": [8, 640, 16],
        },
    ]
    return payload


def _build_fake_results(
    *,
    config,
    failed_backend: str | None = None,
) -> tuple[BackendBenchmarkWorkloadResult, ...]:
    backend_factors = {
        "torch": 1.0,
        "onnxruntime": 0.8,
        "tensorrt": 0.55,
    }
    providers = {
        "torch": None,
        "onnxruntime": "CUDAExecutionProvider",
        "tensorrt": None,
    }
    implementations = {
        "torch": "campp_encoder",
        "onnxruntime": "onnxruntime_session",
        "tensorrt": "tensorrt_plan",
    }

    rows: list[BackendBenchmarkWorkloadResult] = []
    for backend in config.evaluation.backends:
        factor = backend_factors[backend]
        for workload in config.workloads:
            if failed_backend == backend and workload.workload_id == "batch4_long":
                rows.append(
                    BackendBenchmarkWorkloadResult(
                        backend=backend,
                        provider=providers[backend],
                        implementation=implementations[backend],
                        version="1.0.0",
                        device="cuda",
                        workload_id=workload.workload_id,
                        description=workload.description,
                        batch_size=workload.batch_size,
                        frame_count=workload.frame_count,
                        status="failed",
                        initialization_seconds=None,
                        cold_start_seconds=None,
                        warm_mean_latency_ms=None,
                        warm_median_latency_ms=None,
                        warm_p95_latency_ms=None,
                        warm_stddev_latency_ms=None,
                        warm_latency_cv=None,
                        throughput_items_per_second=None,
                        throughput_frames_per_second=None,
                        process_rss_peak_mib=512.0,
                        process_rss_delta_mib=64.0,
                        process_gpu_peak_mib=256.0,
                        process_gpu_delta_mib=48.0,
                        mean_abs_diff=None,
                        max_abs_diff=None,
                        cosine_distance=None,
                        quality_passed=None,
                        error="backend execution failed",
                    )
                )
                continue

            warm_latency_ms = round((workload.frame_count / 10.0) * factor, 6)
            rows.append(
                BackendBenchmarkWorkloadResult(
                    backend=backend,
                    provider=providers[backend],
                    implementation=implementations[backend],
                    version="1.0.0",
                    device="cuda",
                    workload_id=workload.workload_id,
                    description=workload.description,
                    batch_size=workload.batch_size,
                    frame_count=workload.frame_count,
                    status="passed",
                    initialization_seconds=round(0.05 * factor, 6),
                    cold_start_seconds=round(0.09 * factor, 6),
                    warm_mean_latency_ms=warm_latency_ms,
                    warm_median_latency_ms=warm_latency_ms,
                    warm_p95_latency_ms=round(warm_latency_ms * 1.05, 6),
                    warm_stddev_latency_ms=round(warm_latency_ms * 0.03, 6),
                    warm_latency_cv=0.03,
                    throughput_items_per_second=round(
                        workload.batch_size / (warm_latency_ms / 1_000.0),
                        6,
                    ),
                    throughput_frames_per_second=round(
                        (workload.batch_size * workload.frame_count) / (warm_latency_ms / 1_000.0),
                        6,
                    ),
                    process_rss_peak_mib=512.0 + workload.batch_size,
                    process_rss_delta_mib=64.0 + factor,
                    process_gpu_peak_mib=256.0 + workload.frame_count / 10.0,
                    process_gpu_delta_mib=48.0 + factor,
                    mean_abs_diff=0.0 if backend == "torch" else 0.001 * factor,
                    max_abs_diff=0.0 if backend == "torch" else 0.002 * factor,
                    cosine_distance=0.0 if backend == "torch" else 0.0002 * factor,
                    quality_passed=True,
                    error=None,
                )
            )
    return tuple(rows)
