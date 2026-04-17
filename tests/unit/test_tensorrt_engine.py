from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

import kryptonite.serve.tensorrt_engine as tensorrt_engine
import kryptonite.serve.tensorrt_engine_runtime as tensorrt_runtime
from kryptonite.config import load_project_config
from kryptonite.models import CAMPPlusConfig, CAMPPlusEncoder
from kryptonite.serve.onnx_export import (
    CAMPPONNXExportRequest,
    ONNXSmokeValidation,
    export_campp_checkpoint_to_onnx,
)
from kryptonite.serve.tensorrt_engine import (
    TensorRTFP16Profile,
    TensorRTFP16SampleResult,
    build_tensorrt_fp16_report,
    write_tensorrt_fp16_report,
)
from kryptonite.serve.tensorrt_engine_config import load_tensorrt_fp16_config


def test_tensorrt_fp16_workflow_writes_engine_and_promotes_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path, metadata_path, engine_path = _write_tensorrt_config_fixture(
        tmp_path,
        onnxruntime_validated=True,
        monkeypatch=monkeypatch,
    )

    monkeypatch.setattr(
        tensorrt_engine,
        "_build_serialized_tensorrt_engine",
        lambda **_: b"fake-plan-bytes",
    )
    monkeypatch.setattr(
        tensorrt_engine,
        "_validate_tensorrt_engine",
        lambda **_: (
            TensorRTFP16SampleResult(
                sample_id="short",
                profile_id="default",
                batch_size=1,
                frame_count=100,
                output_shape=(1, 32),
                max_abs_diff=0.002,
                mean_abs_diff=0.001,
                cosine_distance=0.0001,
                torch_latency_ms=2.4,
                tensorrt_latency_ms=1.2,
                speedup_ratio=2.0,
                passed_quality=True,
                passed_speedup=True,
            ),
            TensorRTFP16SampleResult(
                sample_id="medium",
                profile_id="default",
                batch_size=1,
                frame_count=200,
                output_shape=(1, 32),
                max_abs_diff=0.003,
                mean_abs_diff=0.0015,
                cosine_distance=0.0002,
                torch_latency_ms=3.6,
                tensorrt_latency_ms=1.8,
                speedup_ratio=2.0,
                passed_quality=True,
                passed_speedup=True,
            ),
        ),
    )

    config = load_tensorrt_fp16_config(config_path=config_path)
    report = build_tensorrt_fp16_report(config, config_path=config_path)
    written = write_tensorrt_fp16_report(report)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert engine_path.read_bytes() == b"fake-plan-bytes"
    assert written.summary.passed is True
    assert written.promotion.applied is True
    assert [profile.profile_id for profile in report.profiles] == ["default"]
    assert report.builder_optimization_level == 3
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()
    assert "`default`: min" in Path(written.report_markdown_path).read_text(encoding="utf-8")
    assert metadata["inference_package"]["artifacts"]["tensorrt_engine_file"] == (
        "artifacts/model-bundle-campp-test/model.plan"
    )
    assert metadata["inference_package"]["validated_backends"]["tensorrt"] is True
    assert metadata["tensorrt_engine_file"] == "artifacts/model-bundle-campp-test/model.plan"
    assert metadata["export_validation"]["tensorrt_fp16_validated"] is True


def test_tensorrt_fp16_workflow_requires_promoted_onnx_parity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path, _, _ = _write_tensorrt_config_fixture(
        tmp_path,
        onnxruntime_validated=False,
        monkeypatch=monkeypatch,
    )

    config = load_tensorrt_fp16_config(config_path=config_path)

    try:
        build_tensorrt_fp16_report(config, config_path=config_path)
    except RuntimeError as exc:
        assert "parity-promoted ONNX bundle" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected build_tensorrt_fp16_report to require ONNX parity.")


def test_load_tensorrt_fp16_config_supports_multiple_profiles(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "release" / "tensorrt-fp16.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'title = "Fixture TensorRT FP16"',
                'report_id = "fixture-tensorrt-fp16"',
                'summary = "Fixture TensorRT FP16 report."',
                'project_root = "."',
                'output_root = "artifacts/release/current/fp16"',
                "",
                "[artifacts]",
                'model_bundle_metadata_path = "artifacts/model-bundle-campp-test/metadata.json"',
                'engine_output_path = "artifacts/model-bundle-campp-test/model.plan"',
                "",
                "[build]",
                "workspace_size_mib = 1024",
                "builder_optimization_level = 5",
                "promote_validated_backend = true",
                "require_onnxruntime_parity = true",
                "",
                "[[build.profiles]]",
                'profile_id = "short"',
                "min_batch_size = 1",
                "opt_batch_size = 1",
                "max_batch_size = 4",
                "min_frame_count = 80",
                "opt_frame_count = 120",
                "max_frame_count = 160",
                "",
                "[[build.profiles]]",
                'profile_id = "mid"',
                "min_batch_size = 1",
                "opt_batch_size = 2",
                "max_batch_size = 4",
                "min_frame_count = 80",
                "opt_frame_count = 256",
                "max_frame_count = 384",
                "",
                "[[build.profiles]]",
                'profile_id = "long"',
                "min_batch_size = 1",
                "opt_batch_size = 1",
                "max_batch_size = 2",
                "min_frame_count = 80",
                "opt_frame_count = 512",
                "max_frame_count = 800",
                "",
                "[evaluation]",
                "seed = 7",
                "warmup_iterations = 1",
                "benchmark_iterations = 2",
                "max_mean_abs_diff = 0.01",
                "max_cosine_distance = 0.001",
                "min_speedup_ratio = 1.05",
                "",
                "[[evaluation.samples]]",
                'sample_id = "short"',
                "batch_size = 1",
                "frame_count = 100",
                "",
                "[[evaluation.samples]]",
                'sample_id = "mid"',
                "batch_size = 1",
                "frame_count = 240",
                "",
                "[[evaluation.samples]]",
                'sample_id = "long"',
                "batch_size = 1",
                "frame_count = 480",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_tensorrt_fp16_config(config_path=config_path)

    assert [profile.profile_id for profile in config.build.profiles] == ["short", "mid", "long"]
    assert config.build.builder_optimization_level == 5
    assert config.build.max_frame_count == 800


def test_load_tensorrt_fp16_config_requires_sample_coverage_by_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "release" / "tensorrt-fp16.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'title = "Fixture TensorRT FP16"',
                'report_id = "fixture-tensorrt-fp16"',
                'summary = "Fixture TensorRT FP16 report."',
                'project_root = "."',
                'output_root = "artifacts/release/current/fp16"',
                "",
                "[artifacts]",
                'model_bundle_metadata_path = "artifacts/model-bundle-campp-test/metadata.json"',
                'engine_output_path = "artifacts/model-bundle-campp-test/model.plan"',
                "",
                "[build]",
                "workspace_size_mib = 1024",
                "",
                "[[build.profiles]]",
                'profile_id = "short"',
                "min_batch_size = 1",
                "opt_batch_size = 1",
                "max_batch_size = 1",
                "min_frame_count = 80",
                "opt_frame_count = 120",
                "max_frame_count = 160",
                "",
                "[[build.profiles]]",
                'profile_id = "long"',
                "min_batch_size = 1",
                "opt_batch_size = 1",
                "max_batch_size = 1",
                "min_frame_count = 300",
                "opt_frame_count = 384",
                "max_frame_count = 480",
                "",
                "[evaluation]",
                "seed = 7",
                "warmup_iterations = 1",
                "benchmark_iterations = 2",
                "max_mean_abs_diff = 0.01",
                "max_cosine_distance = 0.001",
                "min_speedup_ratio = 1.05",
                "",
                "[[evaluation.samples]]",
                'sample_id = "too-long"',
                "batch_size = 1",
                "frame_count = 240",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_tensorrt_fp16_config(config_path=config_path)
    except ValueError as exc:
        assert "not covered by any build.profiles" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected the config loader to reject uncovered evaluation samples.")


def test_select_profile_prefers_smallest_covering_profile() -> None:
    profiles = (
        TensorRTFP16Profile(
            profile_id="short",
            min_shape=(1, 80, 80),
            opt_shape=(1, 120, 80),
            max_shape=(4, 160, 80),
        ),
        TensorRTFP16Profile(
            profile_id="mid",
            min_shape=(1, 80, 80),
            opt_shape=(2, 256, 80),
            max_shape=(4, 384, 80),
        ),
        TensorRTFP16Profile(
            profile_id="long",
            min_shape=(1, 80, 80),
            opt_shape=(1, 512, 80),
            max_shape=(2, 800, 80),
        ),
    )

    assert tensorrt_runtime._select_profile(profiles, shape=(1, 100, 80)).profile_id == "short"
    assert tensorrt_runtime._select_profile(profiles, shape=(1, 240, 80)).profile_id == "mid"
    assert tensorrt_runtime._select_profile(profiles, shape=(1, 480, 80)).profile_id == "long"


def _write_tensorrt_config_fixture(
    tmp_path: Path,
    *,
    onnxruntime_validated: bool,
    monkeypatch,
) -> tuple[Path, Path, Path]:
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

    metadata_path = tmp_path / exported.metadata_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["inference_package"]["validated_backends"]["onnxruntime"] = onnxruntime_validated
    metadata["export_validation"]["runtime_backends_promoted"] = onnxruntime_validated
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config_path = tmp_path / "configs" / "release" / "tensorrt-fp16.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                'title = "Fixture TensorRT FP16"',
                'report_id = "fixture-tensorrt-fp16"',
                'summary = "Fixture TensorRT FP16 report."',
                f'project_root = "{tmp_path.as_posix()}"',
                'output_root = "artifacts/release/current/fp16"',
                "",
                "[artifacts]",
                f'model_bundle_metadata_path = "{exported.metadata_path}"',
                'engine_output_path = "artifacts/model-bundle-campp-test/model.plan"',
                "",
                "[build]",
                "workspace_size_mib = 1024",
                "min_batch_size = 1",
                "opt_batch_size = 1",
                "max_batch_size = 1",
                "min_frame_count = 80",
                "opt_frame_count = 200",
                "max_frame_count = 400",
                "promote_validated_backend = true",
                "require_onnxruntime_parity = true",
                "",
                "[evaluation]",
                "seed = 7",
                "warmup_iterations = 1",
                "benchmark_iterations = 2",
                "max_mean_abs_diff = 0.01",
                "max_cosine_distance = 0.001",
                "min_speedup_ratio = 1.05",
                "",
                "[[evaluation.samples]]",
                'sample_id = "short"',
                "batch_size = 1",
                "frame_count = 100",
                "",
                "[[evaluation.samples]]",
                'sample_id = "medium"',
                "batch_size = 1",
                "frame_count = 200",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        config_path,
        metadata_path,
        tmp_path / "artifacts" / "model-bundle-campp-test" / "model.plan",
    )
