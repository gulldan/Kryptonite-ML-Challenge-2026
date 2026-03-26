from __future__ import annotations

import torch
from torch import nn

from kryptonite.training.baseline_config import BaselineOptimizationConfig
from kryptonite.training.manifest_speaker_data import TrainingBatch
from kryptonite.training.optimization_runtime import (
    build_training_runtime,
    resolve_precision_runtime,
    run_classification_batches,
)


def test_resolve_precision_runtime_falls_back_to_fp32_off_cuda() -> None:
    runtime = resolve_precision_runtime(precision="bf16", device=torch.device("cpu"))

    assert runtime.requested_precision == "bf16"
    assert runtime.effective_precision == "fp32"
    assert runtime.amp_enabled is False
    assert runtime.grad_scaler_enabled is False


def test_build_training_runtime_supports_adamw_and_plateau_scheduler() -> None:
    model = nn.Linear(8, 4)
    classifier = nn.Linear(4, 2)
    runtime = build_training_runtime(
        parameters=[*model.parameters(), *classifier.parameters()],
        optimization_config=BaselineOptimizationConfig(
            optimizer_name="adamw",
            scheduler_name="plateau",
            learning_rate=1e-3,
            min_learning_rate=1e-5,
            gradient_accumulation_steps=3,
            grad_clip_norm=None,
        ),
        precision="fp16",
        device=torch.device("cpu"),
        max_epochs=4,
    )

    assert isinstance(runtime.optimizer, torch.optim.AdamW)
    assert runtime.scheduler.name == "plateau"
    assert runtime.scheduler.uses_metric is True
    assert runtime.gradient_accumulation_steps == 3
    assert runtime.precision.effective_precision == "fp32"

    runtime.step_scheduler(mean_loss=1.0)


def test_run_classification_batches_honors_gradient_accumulation(monkeypatch) -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 4))
    classifier = nn.Linear(4, 2)
    criterion = nn.CrossEntropyLoss()
    runtime = build_training_runtime(
        parameters=[*model.parameters(), *classifier.parameters()],
        optimization_config=BaselineOptimizationConfig(
            optimizer_name="sgd",
            scheduler_name="constant",
            learning_rate=0.05,
            min_learning_rate=0.0,
            gradient_accumulation_steps=2,
            grad_clip_norm=None,
        ),
        precision="fp32",
        device=torch.device("cpu"),
        max_epochs=2,
    )

    step_calls = 0
    original_step = runtime.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(runtime.optimizer, "step", counted_step)

    total_loss, total_correct, total_examples = run_classification_batches(
        model=model,
        classifier=classifier,
        criterion=criterion,
        training_runtime=runtime,
        loader=_build_batches(batch_count=3),
        device=torch.device("cpu"),
    )

    assert total_loss > 0.0
    assert total_correct >= 0
    assert total_examples == 6
    assert step_calls == 2


def _build_batches(*, batch_count: int) -> list[TrainingBatch]:
    batches: list[TrainingBatch] = []
    for batch_index in range(batch_count):
        labels = torch.tensor([batch_index % 2, (batch_index + 1) % 2], dtype=torch.long)
        batches.append(
            TrainingBatch(
                features=torch.randn(2, 8, 4, dtype=torch.float32),
                labels=labels,
                speaker_ids=("speaker_a", "speaker_b"),
                utterance_ids=(f"utt-{batch_index}-0", f"utt-{batch_index}-1"),
                clean_sample_mask=torch.ones(2, dtype=torch.bool),
            )
        )
    return batches
