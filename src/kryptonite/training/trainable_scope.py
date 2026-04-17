"""Helpers for parameter-efficient encoder adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from torch import nn

EncoderTrainableScope = Literal["all", "batchnorm-affine"]

_BATCHNORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
_SCOPE_ALIASES: dict[str, EncoderTrainableScope] = {
    "all": "all",
    "batchnorm": "batchnorm-affine",
    "batchnorm-affine": "batchnorm-affine",
    "batchnorm_affine": "batchnorm-affine",
    "bn": "batchnorm-affine",
    "bn-affine": "batchnorm-affine",
    "bn_only": "batchnorm-affine",
    "bn-only": "batchnorm-affine",
}


@dataclass(frozen=True, slots=True)
class TrainableScopeSummary:
    scope: EncoderTrainableScope
    total_parameters: int
    trainable_parameters: int
    trainable_fraction: float
    trainable_tensors: tuple[str, ...]
    batchnorm_module_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_encoder_trainable_scope(scope: str) -> EncoderTrainableScope:
    normalized = scope.strip().lower()
    try:
        return _SCOPE_ALIASES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(_SCOPE_ALIASES))
        raise ValueError(
            f"Unsupported encoder trainable scope {scope!r}. Supported aliases: {supported}."
        ) from error


def apply_encoder_trainable_scope(
    encoder: nn.Module,
    *,
    scope: str,
) -> TrainableScopeSummary:
    """Apply a trainable-parameter scope to an encoder and return a summary.

    ``batchnorm-affine`` freezes all encoder parameters except BatchNorm affine
    tensors. BatchNorm running statistics still update when the training loop
    calls ``model.train()``.
    """

    normalized_scope = normalize_encoder_trainable_scope(scope)
    batchnorm_module_count = sum(
        1 for module in encoder.modules() if isinstance(module, _BATCHNORM_TYPES)
    )

    if normalized_scope == "all":
        for parameter in encoder.parameters():
            parameter.requires_grad = True
    elif normalized_scope == "batchnorm-affine":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        for module in encoder.modules():
            if not isinstance(module, _BATCHNORM_TYPES) or not module.affine:
                continue
            if module.weight is not None:
                module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True
    else:
        raise AssertionError(f"Unhandled encoder trainable scope: {normalized_scope}")

    named_parameters = tuple(encoder.named_parameters())
    total_parameters = sum(parameter.numel() for _, parameter in named_parameters)
    trainable_tensors = tuple(
        name for name, parameter in named_parameters if parameter.requires_grad
    )
    trainable_parameters = sum(
        parameter.numel() for _, parameter in named_parameters if parameter.requires_grad
    )
    trainable_fraction = 0.0 if total_parameters == 0 else trainable_parameters / total_parameters
    return TrainableScopeSummary(
        scope=normalized_scope,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        trainable_fraction=trainable_fraction,
        trainable_tensors=trainable_tensors,
        batchnorm_module_count=batchnorm_module_count,
    )


__all__ = [
    "EncoderTrainableScope",
    "TrainableScopeSummary",
    "apply_encoder_trainable_scope",
    "normalize_encoder_trainable_scope",
]
