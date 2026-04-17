"""Checkpoint helpers shared by ERes2NetV2 evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kryptonite.deployment import resolve_project_path

from .model import ERes2NetV2Config, ERes2NetV2Encoder

KNOWN_ERES2NETV2_CHECKPOINT_NAMES = ("eres2netv2_encoder.pt",)


def resolve_eres2netv2_checkpoint_path(
    *,
    checkpoint_path: str | Path,
    project_root: str | Path,
) -> Path:
    resolved = resolve_project_path(str(project_root), str(checkpoint_path))
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        for candidate_name in KNOWN_ERES2NETV2_CHECKPOINT_NAMES:
            candidate = resolved / candidate_name
            if candidate.is_file():
                return candidate
        expected = ", ".join(str(resolved / name) for name in KNOWN_ERES2NETV2_CHECKPOINT_NAMES)
        raise FileNotFoundError(
            "ERes2NetV2 run directory does not contain a known checkpoint file. "
            f"Expected one of: {expected}."
        )
    raise FileNotFoundError(
        f"Checkpoint not found at {resolved}. Provide either a checkpoint file or a run directory."
    )


def load_eres2netv2_checkpoint_payload(*, torch: Any, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain an object payload.")
    return dict(payload)


def load_eres2netv2_model_config(payload: object) -> ERes2NetV2Config:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint is missing a `model_config` object.")
    values = dict(payload)
    if isinstance(values.get("num_blocks"), list):
        values["num_blocks"] = tuple(int(item) for item in values["num_blocks"])
    return ERes2NetV2Config(**values)


def require_eres2netv2_state_dict(
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


def load_eres2netv2_encoder_from_checkpoint(
    *,
    torch: Any,
    checkpoint_path: str | Path,
    project_root: str | Path = ".",
) -> tuple[Path, ERes2NetV2Config, Any]:
    resolved_checkpoint_path = resolve_eres2netv2_checkpoint_path(
        checkpoint_path=checkpoint_path,
        project_root=project_root,
    )
    checkpoint_payload = load_eres2netv2_checkpoint_payload(
        torch=torch,
        checkpoint_path=resolved_checkpoint_path,
    )
    model_config = load_eres2netv2_model_config(checkpoint_payload.get("model_config"))
    model_state = require_eres2netv2_state_dict(
        checkpoint_payload.get("model_state_dict"),
        field_name="model_state_dict",
    )
    model = ERes2NetV2Encoder(model_config).to(device="cpu", dtype=torch.float32)
    model.eval()
    model.load_state_dict(model_state)
    return resolved_checkpoint_path, model_config, model


__all__ = [
    "KNOWN_ERES2NETV2_CHECKPOINT_NAMES",
    "load_eres2netv2_checkpoint_payload",
    "load_eres2netv2_encoder_from_checkpoint",
    "load_eres2netv2_model_config",
    "require_eres2netv2_state_dict",
    "resolve_eres2netv2_checkpoint_path",
]
