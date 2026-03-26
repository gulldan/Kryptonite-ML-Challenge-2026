"""Shared optimizer, scheduler, precision, and accumulation runtime helpers."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .baseline_config import BaselineOptimizationConfig
from .manifest_speaker_data import TrainingBatch

logger = logging.getLogger(__name__)

_PRECISION_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
}


def normalize_training_precision(precision: str) -> str:
    normalized = precision.strip().lower()
    try:
        return _PRECISION_ALIASES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(_PRECISION_ALIASES))
        raise ValueError(
            f"Unsupported training precision {precision!r}. Supported values: {supported}."
        ) from error


def validate_training_precision(precision: str, *, baseline_name: str) -> str:
    try:
        return normalize_training_precision(precision)
    except ValueError as error:
        raise ValueError(
            f"{baseline_name} does not support this precision setting: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class PrecisionRuntime:
    requested_precision: str
    effective_precision: str
    amp_enabled: bool
    autocast_dtype: torch.dtype | None
    grad_scaler_enabled: bool

    def autocast_context(self, *, device: torch.device) -> Any:
        if not self.amp_enabled or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=self.autocast_dtype)


@dataclass(slots=True)
class SchedulerRuntime:
    name: str
    scheduler: Any
    uses_metric: bool = False

    def step(self, *, mean_loss: float | None = None) -> None:
        if self.scheduler is None:
            return
        if self.uses_metric:
            if mean_loss is None:
                raise ValueError("plateau scheduler requires mean_loss when stepping.")
            self.scheduler.step(mean_loss)
            return
        self.scheduler.step()


@dataclass(slots=True)
class TrainingOptimizationRuntime:
    optimizer_name: str
    scheduler_name: str
    optimizer: torch.optim.Optimizer
    scheduler: SchedulerRuntime
    precision: PrecisionRuntime
    gradient_accumulation_steps: int
    grad_clip_norm: float | None
    trainable_parameters: tuple[nn.Parameter, ...]
    grad_scaler: torch.cuda.amp.GradScaler | None

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: torch.Tensor) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            return
        loss.backward()

    def step_optimizer(self) -> None:
        if self.grad_scaler is not None:
            if self.grad_clip_norm is not None:
                self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.trainable_parameters, max_norm=self.grad_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            return
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.trainable_parameters, max_norm=self.grad_clip_norm)
        self.optimizer.step()

    def step_scheduler(self, *, mean_loss: float | None = None) -> None:
        self.scheduler.step(mean_loss=mean_loss)

    def current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


def resolve_precision_runtime(*, precision: str, device: torch.device) -> PrecisionRuntime:
    normalized = normalize_training_precision(precision)
    if normalized == "fp32":
        return PrecisionRuntime(
            requested_precision=normalized,
            effective_precision="fp32",
            amp_enabled=False,
            autocast_dtype=None,
            grad_scaler_enabled=False,
        )
    if device.type != "cuda":
        logger.info(
            "Requested %s training precision on %s; falling back to fp32 because AMP is only "
            "enabled on CUDA in this repository runtime.",
            normalized,
            device.type,
        )
        return PrecisionRuntime(
            requested_precision=normalized,
            effective_precision="fp32",
            amp_enabled=False,
            autocast_dtype=None,
            grad_scaler_enabled=False,
        )
    if normalized == "bf16":
        return PrecisionRuntime(
            requested_precision=normalized,
            effective_precision="bf16",
            amp_enabled=True,
            autocast_dtype=torch.bfloat16,
            grad_scaler_enabled=False,
        )
    return PrecisionRuntime(
        requested_precision=normalized,
        effective_precision="fp16",
        amp_enabled=True,
        autocast_dtype=torch.float16,
        grad_scaler_enabled=True,
    )


def build_training_runtime(
    *,
    parameters: list[nn.Parameter],
    optimization_config: BaselineOptimizationConfig,
    precision: str,
    device: torch.device,
    max_epochs: int,
) -> TrainingOptimizationRuntime:
    trainable_parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable_parameters:
        raise ValueError("Training runtime requires at least one trainable parameter.")

    optimizer_name = optimization_config.optimizer_name.strip().lower()
    scheduler_name = optimization_config.scheduler_name.strip().lower()
    optimizer = _build_optimizer(
        parameters=trainable_parameters,
        optimization_config=optimization_config,
        optimizer_name=optimizer_name,
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        optimization_config=optimization_config,
        scheduler_name=scheduler_name,
        max_epochs=max_epochs,
    )
    precision_runtime = resolve_precision_runtime(precision=precision, device=device)
    grad_scaler = (
        torch.cuda.amp.GradScaler(enabled=True) if precision_runtime.grad_scaler_enabled else None
    )
    return TrainingOptimizationRuntime(
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        optimizer=optimizer,
        scheduler=scheduler,
        precision=precision_runtime,
        gradient_accumulation_steps=optimization_config.gradient_accumulation_steps,
        grad_clip_norm=optimization_config.grad_clip_norm,
        trainable_parameters=trainable_parameters,
        grad_scaler=grad_scaler,
    )


def run_classification_batches(
    *,
    model: nn.Module,
    classifier: nn.Module,
    criterion: nn.Module,
    training_runtime: TrainingOptimizationRuntime,
    loader: Any,
    device: torch.device,
) -> tuple[float, int, int]:
    model.train()
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    accumulation_steps = training_runtime.gradient_accumulation_steps
    pending_step = False

    training_runtime.zero_grad()
    for batch_index, batch in enumerate(loader, start=1):
        if not isinstance(batch, TrainingBatch):
            raise TypeError(f"Expected TrainingBatch instances, got {type(batch)!r}.")

        features = batch.features.to(device=device, dtype=torch.float32)
        labels = batch.labels.to(device=device)
        with training_runtime.precision.autocast_context(device=device):
            embeddings = model(features)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            scaled_loss = loss / accumulation_steps
        training_runtime.backward(scaled_loss)
        pending_step = True

        if batch_index % accumulation_steps == 0:
            training_runtime.step_optimizer()
            training_runtime.zero_grad()
            pending_step = False

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    if pending_step:
        training_runtime.step_optimizer()
        training_runtime.zero_grad()

    return total_loss, total_correct, total_examples


def _build_optimizer(
    *,
    parameters: tuple[nn.Parameter, ...],
    optimization_config: BaselineOptimizationConfig,
    optimizer_name: str,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=optimization_config.learning_rate,
            betas=(optimization_config.adam_beta1, optimization_config.adam_beta2),
            eps=optimization_config.adam_epsilon,
            weight_decay=optimization_config.weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=optimization_config.learning_rate,
        momentum=optimization_config.momentum,
        nesterov=True,
        weight_decay=optimization_config.weight_decay,
    )


def _build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    optimization_config: BaselineOptimizationConfig,
    scheduler_name: str,
    max_epochs: int,
) -> SchedulerRuntime:
    if scheduler_name == "constant":
        return SchedulerRuntime(name=scheduler_name, scheduler=None, uses_metric=False)
    if scheduler_name == "plateau":
        return SchedulerRuntime(
            name=scheduler_name,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=optimization_config.plateau_factor,
                patience=optimization_config.plateau_patience_epochs,
                threshold=optimization_config.plateau_threshold,
                min_lr=optimization_config.min_learning_rate,
            ),
            uses_metric=True,
        )
    return SchedulerRuntime(
        name=scheduler_name,
        scheduler=torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_build_cosine_lr_lambda(
                max_epochs=max_epochs,
                warmup_epochs=optimization_config.warmup_epochs,
                learning_rate=optimization_config.learning_rate,
                min_learning_rate=optimization_config.min_learning_rate,
            ),
        ),
        uses_metric=False,
    )


def _build_cosine_lr_lambda(
    *,
    max_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    min_learning_rate: float,
) -> Any:
    min_ratio = min_learning_rate / learning_rate

    def _lambda(epoch_index: int) -> float:
        if warmup_epochs == 0 and epoch_index == 0:
            return 1.0
        current_epoch = epoch_index + 1
        if warmup_epochs > 0 and current_epoch <= warmup_epochs:
            return max(min_ratio, current_epoch / warmup_epochs)
        cosine_steps = max(1, max_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (current_epoch - warmup_epochs) / cosine_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + ((1.0 - min_ratio) * cosine)

    return _lambda


__all__ = [
    "PrecisionRuntime",
    "SchedulerRuntime",
    "TrainingOptimizationRuntime",
    "build_training_runtime",
    "normalize_training_precision",
    "resolve_precision_runtime",
    "run_classification_batches",
    "validate_training_precision",
]
