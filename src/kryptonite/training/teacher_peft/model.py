"""Model, checkpoint, and lazy Hugging Face integration for PEFT speaker runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch import nn

from .config import TeacherPeftAdapterConfig, TeacherPeftModelConfig

KNOWN_TEACHER_PEFT_CHECKPOINT_NAMES = ("teacher_peft", "w2vbert2_sv")


class _LazyModuleAttr:
    def __init__(self, module_name: str, attr_name: str) -> None:
        self._module_name = module_name
        self._attr_name = attr_name
        self._override: Any | None = None

    def __getattr__(self, name: str) -> Any:
        target = self._override
        if target is None:
            module = __import__(self._module_name, fromlist=[self._attr_name])
            target = getattr(module, self._attr_name)
        return getattr(target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_module_name", "_attr_name", "_override"}:
            object.__setattr__(self, name, value)
            return
        if self._override is None:
            module = __import__(self._module_name, fromlist=[self._attr_name])
            self._override = getattr(module, self._attr_name)
        setattr(self._override, name, value)


AutoFeatureExtractor = _LazyModuleAttr("transformers", "AutoFeatureExtractor")
AutoModel = _LazyModuleAttr("transformers", "AutoModel")
PeftModel = _LazyModuleAttr("peft", "PeftModel")


class MaskedAttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.output_dim = input_dim * 2
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
        )

    def forward(self, hidden_state: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if frame_mask.ndim != 2:
            raise ValueError("frame_mask must have shape [batch, frames]")
        logits = self.attention(hidden_state.transpose(1, 2))
        expanded_mask = frame_mask.unsqueeze(1)
        fill_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~expanded_mask, fill_value)
        weights = torch.softmax(logits, dim=-1).transpose(1, 2)
        weights = weights * frame_mask.unsqueeze(-1).to(dtype=weights.dtype)
        normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / normalizer
        mean = torch.sum(hidden_state * weights, dim=1)
        second_moment = torch.sum((hidden_state**2) * weights, dim=1)
        std = torch.sqrt((second_moment - mean.square()).clamp_min(1e-5))
        return torch.cat([mean, std], dim=1)


class TeacherPeftEncoder(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        hidden_size: int,
        embedding_dim: int,
        projection_dropout: float,
        pooling_mode: str = "mean",
        mfa_num_layers: int = 1,
        layer_adapter_enabled: bool = False,
        adapter_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.pooling_mode = pooling_mode
        self.mfa_num_layers = mfa_num_layers
        self.layer_adapter_enabled = layer_adapter_enabled
        self.adapter_dim = adapter_dim
        self.resolved_mfa_layers = (
            resolve_hidden_state_count(backbone) if mfa_num_layers == -1 else mfa_num_layers
        )

        layer_dim = adapter_dim if layer_adapter_enabled else hidden_size
        aggregated_dim = layer_dim * self.resolved_mfa_layers
        self.layer_adapters: nn.ModuleList | None = None
        if layer_adapter_enabled:
            self.layer_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size, adapter_dim),
                        nn.LayerNorm(adapter_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(adapter_dim, adapter_dim),
                    )
                    for _ in range(self.resolved_mfa_layers)
                ]
            )
        if pooling_mode == "mean":
            pooled_dim = aggregated_dim
            self.temporal_pool: nn.Module | None = None
        elif pooling_mode == "asp":
            self.temporal_pool = MaskedAttentiveStatisticsPooling(
                input_dim=aggregated_dim,
                hidden_dim=min(max(aggregated_dim // 2, layer_dim), 512),
            )
            pooled_dim = self.temporal_pool.output_dim
        else:
            raise ValueError("pooling_mode must be one of: mean, asp")

        self.pre_projection_norm = nn.LayerNorm(pooled_dim)
        self.projection = nn.Linear(pooled_dim, embedding_dim)
        self.post_projection_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(projection_dropout)

    def forward(self, **model_inputs: torch.Tensor) -> torch.Tensor:
        require_hidden_states = self.mfa_num_layers != 1 or self.layer_adapter_enabled
        outputs = self.backbone(
            **model_inputs,
            output_hidden_states=require_hidden_states,
            return_dict=True,
        )
        hidden_state = self._aggregate_hidden_states(outputs)
        frame_mask = _resolve_frame_attention_mask(
            backbone=self.backbone,
            attention_mask=model_inputs.get("attention_mask"),
            sequence_length=hidden_state.shape[1],
            device=hidden_state.device,
        )
        if self.pooling_mode == "mean":
            pooled = _masked_mean(hidden_state, frame_mask)
        else:
            assert self.temporal_pool is not None
            pooled = self.temporal_pool(hidden_state, frame_mask)
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

    def _aggregate_hidden_states(self, outputs: Any) -> torch.Tensor:
        if self.mfa_num_layers == 1 and not self.layer_adapter_enabled:
            return outputs.last_hidden_state
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Backbone did not return hidden_states for multi-layer aggregation.")
        selected = tuple(
            hidden_states if self.mfa_num_layers == -1 else hidden_states[-self.mfa_num_layers :]
        )
        if not selected:
            raise RuntimeError("No hidden states were selected for MFA aggregation.")
        if self.layer_adapters is None:
            return torch.cat(selected, dim=-1)
        if len(selected) != len(self.layer_adapters):
            if len(self.layer_adapters) == 1:
                adapters = [self.layer_adapters[0] for _ in range(len(selected))]
            else:
                raise RuntimeError("Layer adapter count does not match selected hidden states.")
        else:
            adapters = list(self.layer_adapters)
        return torch.cat(
            [adapter(layer) for adapter, layer in zip(adapters, selected, strict=True)],
            dim=-1,
        )


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
    from peft import LoraConfig, TaskType, get_peft_model

    backbone = AutoModel.from_pretrained(
        model_config.model_id,
        revision=model_config.revision,
        token=token,
    )
    prepare_teacher_backbone_for_training(
        backbone=backbone,
        model_config=model_config,
        peft_only=True,
    )
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


def prepare_teacher_backbone_for_training(
    *,
    backbone: nn.Module,
    model_config: TeacherPeftModelConfig,
    peft_only: bool,
) -> None:
    gradient_checkpointing_enable = getattr(backbone, "gradient_checkpointing_enable", None)
    if model_config.gradient_checkpointing and callable(gradient_checkpointing_enable):
        gradient_checkpointing_enable()
        enable_input_require_grads = getattr(backbone, "enable_input_require_grads", None)
        if callable(enable_input_require_grads):
            enable_input_require_grads()
    backbone_config = getattr(backbone, "config", None)
    if backbone_config is not None and hasattr(backbone_config, "use_cache"):
        backbone_config.use_cache = False
    if peft_only or model_config.freeze_feature_encoder:
        _freeze_feature_encoder(backbone)
        if peft_only:
            for name, parameter in backbone.named_parameters():
                if "lora_" not in name:
                    parameter.requires_grad = False
    else:
        for parameter in backbone.parameters():
            parameter.requires_grad = True
    backbone.train()


def merge_teacher_lora_backbone(encoder: TeacherPeftEncoder) -> TeacherPeftEncoder:
    merge = getattr(encoder.backbone, "merge_and_unload", None)
    if callable(merge):
        encoder.backbone = merge()
    return encoder


def resolve_hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    for field_name in ("output_hidden_size", "hidden_size"):
        value = getattr(config, field_name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Unable to determine hidden size for the selected teacher backbone.")


def resolve_hidden_state_count(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    for field_name in ("num_hidden_layers", "encoder_layers"):
        value = getattr(config, field_name, None)
        if isinstance(value, int) and value > 0:
            return value + 1
    raise ValueError("Unable to determine hidden-state count for the selected teacher backbone.")


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


def resolve_teacher_peft_checkpoint_path(
    *,
    checkpoint_path: str | Path,
    project_root: str | Path,
) -> Path:
    from kryptonite.deployment import resolve_project_path

    resolved = resolve_project_path(str(project_root), str(checkpoint_path))
    if resolved.is_file() and resolved.name == "checkpoint_metadata.json":
        return resolved.parent
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found at {resolved}. Provide either the checkpoint directory, "
            "the run directory, or checkpoint_metadata.json."
        )
    if (resolved / "checkpoint_metadata.json").is_file():
        return resolved
    candidates = [path.parent for path in resolved.glob("*/checkpoint_metadata.json")]
    if len(candidates) == 1:
        return candidates[0]
    for name in KNOWN_TEACHER_PEFT_CHECKPOINT_NAMES:
        candidate = resolved / name
        if (candidate / "checkpoint_metadata.json").is_file():
            return candidate
    expected = ", ".join(
        str(resolved / name / "checkpoint_metadata.json")
        for name in KNOWN_TEACHER_PEFT_CHECKPOINT_NAMES
    )
    raise FileNotFoundError(
        "Teacher PEFT run directory does not contain a known checkpoint directory. "
        f"Expected one of: {expected}."
    )


def load_teacher_checkpoint_payload(
    *,
    checkpoint_path: str | Path,
    project_root: str | Path = ".",
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    checkpoint_dir = resolve_teacher_peft_checkpoint_path(
        checkpoint_path=checkpoint_path,
        project_root=project_root,
    )
    metadata = json.loads((checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Teacher checkpoint metadata must be a JSON object.")
    payload = torch.load(checkpoint_dir / "heads.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Teacher head checkpoint must be an object payload.")
    return checkpoint_dir, metadata, payload


def load_teacher_peft_encoder_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    project_root: str | Path = ".",
    token: str | None = None,
    trainable: bool = False,
) -> tuple[Path, dict[str, Any], Any, TeacherPeftEncoder]:
    checkpoint_dir, metadata, payload = load_teacher_checkpoint_payload(
        checkpoint_path=checkpoint_path,
        project_root=project_root,
    )
    model_payload = metadata.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("Teacher checkpoint metadata is missing the `model` section.")
    checkpoint_format = str(metadata.get("checkpoint_format", "peft_adapter"))
    model_config = TeacherPeftModelConfig(**_filtered_model_kwargs(model_payload))
    feature_extractor_dir = checkpoint_dir / "feature_extractor"
    backbone_dir = checkpoint_dir / "adapter"

    if checkpoint_format == "peft_adapter":
        backbone = AutoModel.from_pretrained(
            model_config.model_id,
            revision=model_config.revision,
            token=token,
        )
        backbone = PeftModel.from_pretrained(backbone, backbone_dir, is_trainable=trainable)
    elif checkpoint_format == "full_model":
        backbone = AutoModel.from_pretrained(backbone_dir, token=token)
    else:
        raise ValueError(f"Unsupported teacher checkpoint_format: {checkpoint_format}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_dir, token=token)
    encoder = TeacherPeftEncoder(
        backbone=backbone,
        hidden_size=resolve_hidden_size(backbone),
        embedding_dim=model_config.embedding_dim,
        projection_dropout=model_config.projection_dropout,
        pooling_mode=model_config.pooling_mode,
        mfa_num_layers=model_config.mfa_num_layers,
        layer_adapter_enabled=model_config.layer_adapter_enabled,
        adapter_dim=model_config.adapter_dim,
    )
    encoder_state = payload.get("encoder_head_state_dict")
    if not isinstance(encoder_state, dict):
        raise ValueError("Teacher head checkpoint is missing `encoder_head_state_dict`.")
    encoder.load_state_dict(dict(encoder_state), strict=False)
    encoder.eval()
    return checkpoint_dir, metadata, feature_extractor, encoder


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
    backbone_dir = checkpoint_dir / "adapter"
    feature_extractor_dir = checkpoint_dir / "feature_extractor"
    backbone_dir.mkdir(parents=True, exist_ok=True)
    feature_extractor_dir.mkdir(parents=True, exist_ok=True)

    save_pretrained = getattr(encoder.backbone, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("Teacher backbone must implement save_pretrained().")
    save_pretrained(backbone_dir)
    save_feature_extractor = getattr(feature_extractor, "save_pretrained", None)
    if not callable(save_feature_extractor):
        raise TypeError("Teacher feature extractor must implement save_pretrained().")
    save_feature_extractor(feature_extractor_dir)

    checkpoint_format = "peft_adapter" if _is_peft_backbone(encoder.backbone) else "full_model"
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
                "checkpoint_format": checkpoint_format,
                "model": asdict(model_config),
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
        *backbone_dir.rglob("*"),
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


def _is_peft_backbone(backbone: nn.Module) -> bool:
    peft_model_class = _resolve_peft_model_class()
    return peft_model_class is not None and isinstance(backbone, peft_model_class)


def _resolve_peft_model_class() -> type[nn.Module] | None:
    target: Any = PeftModel
    if isinstance(target, _LazyModuleAttr):
        try:
            module = __import__(target._module_name, fromlist=[target._attr_name])
            target = getattr(module, target._attr_name)
        except Exception:
            return None
    return target if isinstance(target, type) else None


def _filtered_model_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "model_id",
        "feature_extractor_id",
        "revision",
        "embedding_dim",
        "projection_dropout",
        "pooling_mode",
        "gradient_checkpointing",
        "freeze_feature_encoder",
        "mfa_num_layers",
        "layer_adapter_enabled",
        "adapter_dim",
    }
    return {key: value for key, value in values.items() if key in allowed}
