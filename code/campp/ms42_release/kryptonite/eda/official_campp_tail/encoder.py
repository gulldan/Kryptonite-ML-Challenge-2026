"""Encoder backends for the official CAM++ tail pipeline.

This vendored runtime keeps only the torch backend. The TensorRT backend from
the upstream competition repo depends on NVIDIA-specific build/runtime modules
that are intentionally not carried into this repository.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from kryptonite.models.campp.checkpoint import load_campp_encoder_from_checkpoint

from .config import OfficialCamPPTailConfig

EncoderCallable = Callable[[Any], np.ndarray]


def build_encoder(config: OfficialCamPPTailConfig) -> EncoderCallable:
    if config.encoder_backend == "torch":
        return _build_torch_encoder(config)
    if config.encoder_backend == "tensorrt":
        raise ValueError(
            "TensorRT backend is not packaged in this repository. Use --encoder-backend=torch."
        )
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


__all__ = ["EncoderCallable", "build_encoder"]
