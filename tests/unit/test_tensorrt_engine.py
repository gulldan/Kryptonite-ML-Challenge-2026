from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

import kryptonite.serve.tensorrt_engine as tensorrt_engine
from kryptonite.config import load_project_config
from kryptonite.models import CAMPPlusConfig, CAMPPlusEncoder
from kryptonite.serve.onnx_export import CAMPPONNXExportRequest, export_campp_checkpoint_to_onnx
from kryptonite.serve.tensorrt_engine import (
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
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()
    assert metadata["inference_package"]["artifacts"]["tensorrt_engine_file"] == (
        "artifacts/model-bundle-campp-test/model.plan"
    )
    assert metadata["inference_package"]["validated_backends"]["tensorrt"] is True
    assert metadata["tensorrt_engine_file"] == "artifacts/model-bundle-campp-test/model.plan"
    assert metadata["export_validation"]["tensorrt_fp16_validated"] is True


def test_tensorrt_fp16_workflow_requires_promoted_onnx_parity(tmp_path: Path) -> None:
    config_path, _, _ = _write_tensorrt_config_fixture(
        tmp_path,
        onnxruntime_validated=False,
    )

    config = load_tensorrt_fp16_config(config_path=config_path)

    try:
        build_tensorrt_fp16_report(config, config_path=config_path)
    except RuntimeError as exc:
        assert "parity-promoted ONNX bundle" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected build_tensorrt_fp16_report to require ONNX parity.")


def _write_tensorrt_config_fixture(
    tmp_path: Path,
    *,
    onnxruntime_validated: bool,
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
