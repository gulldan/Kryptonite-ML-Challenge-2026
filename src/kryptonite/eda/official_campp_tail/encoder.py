"""Encoder backends for the official CAM++ tail pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from kryptonite.deployment import resolve_project_path
from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint
from kryptonite.runtime.export_boundary import load_export_boundary_from_model_metadata
from kryptonite.runtime.tensorrt_engine_config import load_tensorrt_fp16_config
from kryptonite.runtime.tensorrt_engine_models import TensorRTFP16Profile
from kryptonite.runtime.tensorrt_engine_runtime import _select_profile, _TensorRTEngineRunner

from .config import OfficialCamPPTailConfig

EncoderCallable = Callable[[Any], np.ndarray]


def build_encoder(config: OfficialCamPPTailConfig) -> EncoderCallable:
    if config.encoder_backend == "torch":
        return _build_torch_encoder(config)
    if config.encoder_backend == "tensorrt":
        return _build_tensorrt_encoder(config)
    raise ValueError(f"Unsupported encoder backend={config.encoder_backend!r}")


def _build_torch_encoder(config: OfficialCamPPTailConfig) -> EncoderCallable:
    import torch

    _, _, model = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=config.checkpoint_path,
    )
    device = torch.device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    model = model.to(device)
    model.eval()

    def encode(batch: Any) -> np.ndarray:
        with torch.no_grad():
            return model(batch.to(device)).detach().cpu().numpy()

    return encode


def _build_tensorrt_encoder(config: OfficialCamPPTailConfig) -> EncoderCallable:
    if not config.tensorrt_config:
        raise ValueError("--tensorrt-config is required when --encoder-backend=tensorrt.")

    import torch

    tensorrt_config = load_tensorrt_fp16_config(config_path=config.tensorrt_config)
    project_root = resolve_project_path(tensorrt_config.project_root, ".")
    metadata_path = resolve_project_path(
        str(project_root),
        tensorrt_config.artifacts.model_bundle_metadata_path,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = load_export_boundary_from_model_metadata(metadata)
    feature_dim = _require_static_axis_size(contract.input_tensor.axes[-1].size, "mel_bins")
    profiles = tuple(
        TensorRTFP16Profile(
            profile_id=profile.profile_id,
            min_shape=(profile.min_batch_size, profile.min_frame_count, feature_dim),
            opt_shape=(profile.opt_batch_size, profile.opt_frame_count, feature_dim),
            max_shape=(profile.max_batch_size, profile.max_frame_count, feature_dim),
        )
        for profile in tensorrt_config.build.profiles
    )
    max_batch_size = max(profile.max_shape[0] for profile in profiles)
    if config.batch_size > max_batch_size:
        raise ValueError(
            f"--batch-size={config.batch_size} exceeds the TensorRT max profile batch size "
            f"{max_batch_size}."
        )
    engine_path = (
        resolve_project_path(str(project_root), config.tensorrt_engine_path)
        if config.tensorrt_engine_path
        else resolve_project_path(str(project_root), tensorrt_config.artifacts.engine_output_path)
    )
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("TensorRT encoder backend requires a CUDA device.")
    if device.index is not None:
        torch.cuda.set_device(device)
    runner = _TensorRTEngineRunner(
        engine_path=engine_path,
        input_name=contract.input_tensor.name,
        output_name=contract.output_tensor.name,
    )
    config.resolved_tensorrt_engine_path = str(engine_path)
    config.resolved_tensorrt_profile_ids = [profile.profile_id for profile in profiles]
    print(
        f"[official-campp] TensorRT encoder engine={engine_path} "
        f"profiles={[profile.profile_id for profile in profiles]}",
        flush=True,
    )

    def encode(batch: Any) -> np.ndarray:
        shape = (int(batch.shape[0]), int(batch.shape[1]), int(batch.shape[2]))
        if len(shape) != 3 or shape[-1] != feature_dim:
            raise ValueError(f"Unexpected TensorRT input shape {shape}; feature_dim={feature_dim}.")
        profile = _select_profile(profiles, shape=shape)
        with torch.inference_mode():
            output = runner.run(
                batch.to(device=device, dtype=torch.float32),
                profile_index=profiles.index(profile),
            )
        return output.detach().cpu().float().numpy()

    return encode


def _require_static_axis_size(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be static for TensorRT CAM++ extraction.")
    return value


__all__ = ["EncoderCallable", "build_encoder"]
