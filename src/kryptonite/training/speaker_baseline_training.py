"""Training loop helpers for manifest-backed speaker baselines."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kryptonite.config import ChunkingConfig, ProjectConfig
from kryptonite.deployment import resolve_project_path
from kryptonite.features import UtteranceChunkingRequest

from .baseline_config import BaselineObjectiveConfig, BaselineOptimizationConfig
from .manifest_speaker_data import TrainingBatch
from .optimization_runtime import TrainingOptimizationRuntime, run_classification_batches
from .speaker_baseline_types import (
    EarlyStoppingSummary,
    EpochAwareBatchSampler,
    EpochAwareDataset,
    EpochSummary,
    TrainingLoopResult,
)


def prepare_demo_artifacts_if_needed(
    *,
    project: ProjectConfig,
    train_manifest: str,
    dev_manifest: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    project_root = resolve_project_path(project.paths.project_root, ".")
    resolved_train_manifest = resolve_project_path(str(project_root), train_manifest)
    resolved_dev_manifest = resolve_project_path(str(project_root), dev_manifest)
    if resolved_train_manifest.exists() and resolved_dev_manifest.exists():
        return
    raise FileNotFoundError(
        f"Training manifests not found: {resolved_train_manifest}, {resolved_dev_manifest}"
    )


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)


def build_fixed_train_chunking_request(
    *,
    chunking: ChunkingConfig,
    baseline_name: str,
) -> UtteranceChunkingRequest:
    if chunking.train_num_crops != 1:
        raise ValueError(f"{baseline_name} baseline requires chunking.train_num_crops=1.")
    if not math.isclose(
        chunking.train_min_crop_seconds,
        chunking.train_max_crop_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"{baseline_name} baseline requires fixed-size train crops; set "
            "chunking.train_min_crop_seconds == chunking.train_max_crop_seconds."
        )
    return UtteranceChunkingRequest.from_config(chunking)


def train_epochs(
    *,
    model: nn.Module,
    classifier: nn.Module,
    criterion: nn.Module,
    training_runtime: TrainingOptimizationRuntime,
    loader: Iterable[TrainingBatch],
    dataset: EpochAwareDataset,
    sampler: EpochAwareBatchSampler | None,
    device: torch.device,
    max_epochs: int,
    objective_config: BaselineObjectiveConfig | None = None,
    optimization_config: BaselineOptimizationConfig,
    tracker_run: Any | None,
) -> TrainingLoopResult:
    summaries: list[EpochSummary] = []
    early_stopping_enabled = optimization_config.early_stopping_enabled
    monitor = optimization_config.early_stopping_monitor.strip().lower()
    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs = 0
    stop_reason: str | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    best_classifier_state_dict: dict[str, torch.Tensor] | None = None
    for epoch in range(max_epochs):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss, total_correct, total_examples = run_classification_batches(
            model=model,
            classifier=classifier,
            criterion=criterion,
            training_runtime=training_runtime,
            loader=loader,
            device=device,
            progress_label=f"epoch={epoch + 1}/{max_epochs}",
            objective_config=objective_config,
            pseudo_confidence_threshold=_pseudo_gll_threshold(
                objective_config=objective_config,
                epoch=epoch,
                max_epochs=max_epochs,
            ),
        )

        if total_examples == 0:
            raise ValueError("Training loader produced zero examples.")

        learning_rate = round(training_runtime.current_learning_rate(), 8)
        mean_loss = round(total_loss / total_examples, 6)
        accuracy = round(total_correct / total_examples, 6)
        metric_value = _early_stopping_metric(
            mean_loss=mean_loss,
            accuracy=accuracy,
            monitor=monitor,
        )
        is_best_checkpoint = False
        if early_stopping_enabled:
            if best_value is None or _early_stopping_improved(
                candidate=metric_value,
                best=best_value,
                monitor=monitor,
                min_delta=optimization_config.early_stopping_min_delta,
            ):
                best_value = metric_value
                best_epoch = epoch + 1
                bad_epochs = 0
                is_best_checkpoint = True
                if optimization_config.early_stopping_restore_best:
                    best_model_state_dict = _snapshot_module_state_dict(model)
                    best_classifier_state_dict = _snapshot_module_state_dict(classifier)
            else:
                bad_epochs += 1

        summary = EpochSummary(
            epoch=epoch + 1,
            mean_loss=mean_loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            is_best_checkpoint=is_best_checkpoint,
        )
        summaries.append(summary)
        if tracker_run is not None:
            tracker_run.log_metrics(
                {
                    "train_loss": summary.mean_loss,
                    "train_accuracy": summary.accuracy,
                    "learning_rate": summary.learning_rate,
                },
                step=summary.epoch,
            )
        training_runtime.step_scheduler(mean_loss=summary.mean_loss)

        if (
            early_stopping_enabled
            and summary.epoch >= optimization_config.early_stopping_min_epochs
        ):
            stop_reason = _resolve_early_stopping_reason(
                summary=summary,
                bad_epochs=bad_epochs,
                patience_epochs=optimization_config.early_stopping_patience_epochs,
                stop_train_accuracy=optimization_config.early_stopping_stop_train_accuracy,
            )
            if stop_reason is not None:
                print(
                    "[train] early stopping "
                    f"epoch={summary.epoch} reason={stop_reason} "
                    f"best_epoch={best_epoch} best_{monitor}={best_value}",
                    flush=True,
                )
                break

    early_stopping: EarlyStoppingSummary | None = None
    if early_stopping_enabled:
        early_stopping = EarlyStoppingSummary(
            enabled=True,
            monitor=monitor,
            min_delta=optimization_config.early_stopping_min_delta,
            patience_epochs=optimization_config.early_stopping_patience_epochs,
            min_epochs=optimization_config.early_stopping_min_epochs,
            restore_best=optimization_config.early_stopping_restore_best,
            stop_train_accuracy=optimization_config.early_stopping_stop_train_accuracy,
            stopped=stop_reason is not None,
            reason=stop_reason,
            best_epoch=best_epoch,
            best_value=best_value,
        )
    return TrainingLoopResult(
        summaries=summaries,
        early_stopping=early_stopping,
        best_model_state_dict=best_model_state_dict,
        best_classifier_state_dict=best_classifier_state_dict,
    )


def _pseudo_gll_threshold(
    *,
    objective_config: BaselineObjectiveConfig | None,
    epoch: int,
    max_epochs: int,
) -> float | None:
    if objective_config is None or not objective_config.pseudo_gll_enabled:
        return None
    if max_epochs <= 1:
        return objective_config.pseudo_gll_threshold_end
    progress = epoch / max(max_epochs - 1, 1)
    start = objective_config.pseudo_gll_threshold_start
    end = objective_config.pseudo_gll_threshold_end
    return start + ((end - start) * progress)


def _early_stopping_metric(*, mean_loss: float, accuracy: float, monitor: str) -> float:
    if monitor == "train_accuracy":
        return accuracy
    return mean_loss


def _early_stopping_improved(
    *,
    candidate: float,
    best: float,
    monitor: str,
    min_delta: float,
) -> bool:
    if monitor == "train_accuracy":
        return candidate > best + min_delta
    return candidate < best - min_delta


def _resolve_early_stopping_reason(
    *,
    summary: EpochSummary,
    bad_epochs: int,
    patience_epochs: int,
    stop_train_accuracy: float | None,
) -> str | None:
    if stop_train_accuracy is not None and summary.accuracy >= stop_train_accuracy:
        return "train_accuracy_threshold"
    if bad_epochs > 0 and bad_epochs >= patience_epochs:
        return "patience_exhausted"
    return None


def _snapshot_module_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def write_checkpoint(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    classifier: nn.Module,
    model_config: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    speaker_to_index: Mapping[str, int],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "model_config": dict(model_config),
            "baseline_config": dict(baseline_config),
            "speaker_to_index": dict(speaker_to_index),
        },
        checkpoint_path,
    )
