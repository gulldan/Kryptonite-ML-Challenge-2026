"""Model, batching, and checkpoint helpers for teacher-style PEFT runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel

from .config import TeacherPeftAdapterConfig, TeacherPeftModelConfig


class TeacherPeftEncoder(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        hidden_size: int,
        embedding_dim: int,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pre_projection_norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, embedding_dim)
        self.post_projection_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(projection_dropout)

    def forward(self, **model_inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_state = outputs.last_hidden_state
        frame_mask = _resolve_frame_attention_mask(
            backbone=self.backbone,
            attention_mask=model_inputs.get("attention_mask"),
            sequence_length=hidden_state.shape[1],
            device=hidden_state.device,
        )
        pooled = _masked_mean(hidden_state, frame_mask)
        pooled = self.pre_projection_norm(pooled)
        projected = self.projection(self.dropout(pooled))
        projected = self.post_projection_norm(projected)
        return torch_functional.normalize(projected, dim=1)

    def non_backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }


def load_teacher_feature_extractor(
    *,
    model_config: TeacherPeftModelConfig,
    token: str | None,
) -> Any:
    return AutoFeatureExtractor.from_pretrained(
        model_config.resolved_feature_extractor_id,
        revision=model_config.revision,
        token=token,
    )


def build_teacher_peft_backbone(
    *,
    model_config: TeacherPeftModelConfig,
    adapter_config: TeacherPeftAdapterConfig,
    token: str | None,
) -> nn.Module:
    backbone = AutoModel.from_pretrained(
        model_config.model_id,
        revision=model_config.revision,
        token=token,
    )
    if model_config.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
        enable_input_require_grads = getattr(backbone, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            enable_input_require_grads()
    if model_config.freeze_feature_encoder:
        _freeze_feature_encoder(backbone)
    if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=adapter_config.rank,
        target_modules=_normalize_target_modules(adapter_config.target_modules),
        lora_alpha=adapter_config.alpha,
        lora_dropout=adapter_config.dropout,
        bias=adapter_config.bias,
        use_rslora=adapter_config.use_rslora,
    )
    return get_peft_model(backbone, peft_config)


def resolve_hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    for field_name in ("output_hidden_size", "hidden_size"):
        value = getattr(config, field_name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Unable to determine hidden size for the selected teacher backbone.")


def build_feature_batch(
    *,
    feature_extractor: Any,
    waveforms: Sequence[torch.Tensor],
    sample_rate_hz: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = feature_extractor(
        [
            np.asarray(waveform.detach().cpu(), dtype=np.float32).reshape(-1)
            for waveform in waveforms
        ],
        sampling_rate=sample_rate_hz,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device=device) for key, value in batch.items()}


def write_teacher_checkpoint(
    *,
    checkpoint_dir: Path,
    encoder: TeacherPeftEncoder,
    classifier: nn.Module,
    feature_extractor: Any,
    model_config: TeacherPeftModelConfig,
    adapter_config: TeacherPeftAdapterConfig,
    baseline_config: dict[str, Any],
    speaker_to_index: dict[str, int],
) -> list[Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / "adapter"
    feature_extractor_dir = checkpoint_dir / "feature_extractor"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    feature_extractor_dir.mkdir(parents=True, exist_ok=True)

    save_pretrained = getattr(encoder.backbone, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("Teacher backbone must implement save_pretrained().")
    save_pretrained(adapter_dir)
    save_feature_extractor = getattr(feature_extractor, "save_pretrained", None)
    if not callable(save_feature_extractor):
        raise TypeError("Teacher feature extractor must implement save_pretrained().")
    save_feature_extractor(feature_extractor_dir)

    head_state_path = checkpoint_dir / "heads.pt"
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    torch.save(
        {
            "encoder_head_state_dict": encoder.non_backbone_state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "speaker_to_index": dict(speaker_to_index),
        },
        head_state_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_id": model_config.model_id,
                    "feature_extractor_id": model_config.resolved_feature_extractor_id,
                    "revision": model_config.revision,
                    "embedding_dim": model_config.embedding_dim,
                },
                "adapter": {
                    "rank": adapter_config.rank,
                    "alpha": adapter_config.alpha,
                    "dropout": adapter_config.dropout,
                    "target_modules": list(adapter_config.target_modules),
                    "bias": adapter_config.bias,
                    "use_rslora": adapter_config.use_rslora,
                },
                "baseline_config": baseline_config,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [
        *adapter_dir.rglob("*"),
        *feature_extractor_dir.rglob("*"),
        head_state_path,
        metadata_path,
    ]


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def _normalize_target_modules(target_modules: tuple[str, ...]) -> list[str] | str:
    if len(target_modules) == 1 and target_modules[0] == "all-linear":
        return "all-linear"
    return list(target_modules)


def _freeze_feature_encoder(backbone: nn.Module) -> None:
    for method_name in ("freeze_feature_encoder", "freeze_feature_extractor"):
        method = getattr(backbone, method_name, None)
        if callable(method):
            method()
            return


def _resolve_frame_attention_mask(
    *,
    backbone: nn.Module,
    attention_mask: torch.Tensor | None,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones((1, sequence_length), dtype=torch.bool, device=device)
    if attention_mask.shape[1] == sequence_length:
        return attention_mask.to(device=device, dtype=torch.bool)
    mask_builder = getattr(backbone, "_get_feature_vector_attention_mask", None)
    if callable(mask_builder):
        return mask_builder(sequence_length, attention_mask.to(device=device)).to(dtype=torch.bool)
    return attention_mask[:, :sequence_length].to(device=device, dtype=torch.bool)


def _masked_mean(hidden_state: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    weights = frame_mask.unsqueeze(-1).to(dtype=hidden_state.dtype)
    total = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_state * weights).sum(dim=1) / total
