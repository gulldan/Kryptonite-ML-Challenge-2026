from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from kryptonite.config import load_project_config
from kryptonite.models import CAMPPlusConfig, CAMPPlusEncoder
from kryptonite.models.campp.layers import ContextAwareMaskingLayer
from kryptonite.serve.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.serve.onnx_export import (
    CAMPPONNXExportRequest,
    export_campp_checkpoint_to_onnx,
)


def test_campp_segment_pooling_matches_legacy_expansion() -> None:
    layer = ContextAwareMaskingLayer(
        8,
        4,
        3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    )

    for frames in (80, 99, 100, 101, 250):
        inputs = torch.randn(2, 8, frames)
        pooled = torch.nn.functional.avg_pool1d(
            inputs,
            kernel_size=100,
            stride=100,
            ceil_mode=True,
        )
        shape = pooled.shape
        legacy = (
            pooled.unsqueeze(-1)
            .expand(*shape, 100)
            .reshape(*shape[:-1], -1)[..., : inputs.shape[-1]]
        )
        current = layer.segment_pooling(inputs)

        assert torch.allclose(current, legacy)


def test_export_campp_checkpoint_to_onnx_writes_export_bundle(tmp_path: Path) -> None:
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

    config = load_project_config(
        config_path=Path("configs/base.toml"),
        overrides=[
            f'paths.project_root="{tmp_path.as_posix()}"',
            "features.num_mel_bins=16",
            "tracking.enabled=false",
            "runtime.num_workers=0",
        ],
    )

    exported = export_campp_checkpoint_to_onnx(
        config=config,
        request=CAMPPONNXExportRequest(
            checkpoint_path=str(checkpoint_dir),
            output_root="artifacts/model-bundle-campp-test",
            sample_frame_count=120,
        ),
    )

    model_path = tmp_path / exported.model_path
    metadata_path = tmp_path / exported.metadata_path
    boundary_path = tmp_path / exported.export_boundary_path
    report_json_path = tmp_path / exported.report_json_path
    report_markdown_path = tmp_path / exported.report_markdown_path

    assert model_path.is_file()
    assert metadata_path.is_file()
    assert boundary_path.is_file()
    assert report_json_path.is_file()
    assert report_markdown_path.is_file()
    assert exported.validation.checker_passed is True
    assert exported.validation.onnxruntime_smoke_passed is True
    assert exported.validation.sample_input_shape == (1, 120, 16)
    assert exported.validation.sample_output_shape == (1, 32)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = load_export_boundary_from_model_metadata(metadata)
    report_payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    report_markdown = report_markdown_path.read_text(encoding="utf-8")

    assert metadata["inferencer_backend"] == "campp_encoder"
    assert (
        metadata["source_checkpoint_path"] == "artifacts/baselines/campp/run-001/campp_encoder.pt"
    )
    assert metadata["inference_package"]["validated_backends"] == {
        "onnxruntime": False,
        "tensorrt": False,
        "torch": False,
    }
    assert metadata["export_validation"]["runtime_backends_promoted"] is False
    assert contract.dynamic_time_axis is True
    assert contract.output_tensor.axes[-1].size == 32
    assert report_payload["status"] == "pass"
    assert report_payload["bundle"]["model_version"] == exported.model_version
    assert (
        "Runtime backend promotion stays blocked until the broader parity work in `KVA-539`."
        in report_markdown
    )
