"""Checkpoint helpers shared by CAM++ export and runtime parity workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kryptonite.deployment import resolve_project_path

from .model import CAMPPlusConfig, CAMPPlusEncoder

KNOWN_CAMPP_CHECKPOINT_NAMES = (
    "campp_stage3_encoder.pt",
    "campp_stage2_encoder.pt",
    "campp_stage1_encoder.pt",
    "campp_encoder.pt",
)


def resolve_campp_checkpoint_path(
    *,
    checkpoint_path: str | Path,
    project_root: str | Path,
) -> Path:
    resolved = resolve_project_path(str(project_root), str(checkpoint_path))
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        for candidate_name in KNOWN_CAMPP_CHECKPOINT_NAMES:
            candidate = resolved / candidate_name
            if candidate.is_file():
                return candidate
        expected = ", ".join(str(resolved / name) for name in KNOWN_CAMPP_CHECKPOINT_NAMES)
        raise FileNotFoundError(
            "CAM++ run directory does not contain a known checkpoint file. "
            f"Expected one of: {expected}."
        )
    raise FileNotFoundError(
        f"Checkpoint not found at {resolved}. Provide either a checkpoint file or a run directory."
    )


def load_campp_checkpoint_payload(*, torch: Any, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain an object payload.")
    return dict(payload)


def load_campp_model_config(payload: object) -> CAMPPlusConfig:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint is missing a `model_config` object.")
    values = dict(payload)
    for field_name in (
        "head_res_blocks",
        "block_layers",
        "block_kernel_sizes",
        "block_dilations",
    ):
        value = values.get(field_name)
        if isinstance(value, list):
            values[field_name] = tuple(int(item) for item in value)
    return CAMPPlusConfig(**values)


def require_campp_state_dict(
    payload: object, *, field_name: str = "model_state_dict"
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is missing `{field_name}`.")
    state = dict(payload)
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} must contain string keys.")
        if type(value).__name__ != "Tensor":
            raise ValueError(f"{field_name} must contain torch.Tensor values.")
    return state


def load_campp_encoder_from_checkpoint(
    *,
    torch: Any,
    checkpoint_path: str | Path,
    project_root: str | Path = ".",
) -> tuple[Path, CAMPPlusConfig, Any]:
    resolved_checkpoint_path = resolve_campp_checkpoint_path(
        checkpoint_path=checkpoint_path,
        project_root=project_root,
    )
    checkpoint_payload = load_campp_checkpoint_payload(
        torch=torch,
        checkpoint_path=resolved_checkpoint_path,
    )
    model_config = load_campp_model_config(checkpoint_payload.get("model_config"))
    model_state = require_campp_state_dict(
        checkpoint_payload.get("model_state_dict"),
        field_name="model_state_dict",
    )
    model = CAMPPlusEncoder(model_config).to(device="cpu", dtype=torch.float32)
    model.eval()
    model.load_state_dict(model_state)
    return resolved_checkpoint_path, model_config, model


__all__ = [
    "KNOWN_CAMPP_CHECKPOINT_NAMES",
    "load_campp_checkpoint_payload",
    "load_campp_encoder_from_checkpoint",
    "load_campp_model_config",
    "require_campp_state_dict",
    "resolve_campp_checkpoint_path",
]
